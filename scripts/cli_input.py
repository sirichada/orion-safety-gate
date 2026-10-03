"""
Collects the CLI's run inputs (capabilities, model id, thresholds file) from a
capabilities file, from flags, or, in a real terminal, by asking for whatever
is missing. Running the suite is not this module's concern.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from capabilities import ALLOWED_LANGUAGES, Capabilities, load_capabilities, parse_capabilities
from orchestrator import REPO_ROOT, SuiteError

DEFAULT_THRESHOLDS = REPO_ROOT / "config" / "thresholds.json"
LANGUAGE_NAMES = {"en": "English", "th": "Thai"}
CHECKBOX_HINT = "(up/down arrows to move, Space to tick or untick, Enter to confirm)"
POINTER = "❯"
ACCENT = "#d97757"


class Cancelled(Exception):
    """The user cancelled at a prompt; nothing was run."""


@dataclass(frozen=True)
class Inputs:
    capabilities: Capabilities
    model_id: str
    thresholds_path: Path
    source: str  # "file" | "flags" | "interactive" (anything asked counts as interactive)


def suggest_model_id(model: str) -> str:
    last = Path(model.rstrip("/\\")).name or model
    last = last.rsplit("/", 1)[-1]
    return re.sub(r"[^A-Za-z0-9._-]+", "-", last).strip("-").lower() or "model"


class QuestionaryAsk:
    """Default prompt backend; questionary is imported only when a prompt is actually shown."""

    def __init__(self):
        import questionary

        self.q = questionary
        # Only the cursor row is coloured; ticked rows stay plain, so "where am I" and
        # "what is ticked" never look alike. questionary styles a ticked row as
        # "selected" even under the cursor, so there the pointer alone marks the cursor.
        self.style = questionary.Style(
            [
                ("qmark", f"fg:{ACCENT}"),
                ("pointer", f"fg:{ACCENT} bold"),
                ("highlighted", f"fg:{ACCENT} bold"),
                ("selected", "noreverse"),
                ("instruction", "fg:#888888"),
            ]
        )

    def checkbox(self, message, choices, validate=None):
        # questionary rejects validate=None, so only pass it when there is one.
        extra = {"validate": validate} if validate is not None else {}
        return self.q.checkbox(
            message, choices=choices, instruction=CHECKBOX_HINT, pointer=POINTER, style=self.style, **extra
        ).ask()

    def text(self, message, default=""):
        return self.q.text(message, default=default, style=self.style).ask()

    def confirm(self, message, default=False):
        return self.q.confirm(message, default=default, style=self.style).ask()

    def choice(self, title, value, checked=False, disabled=None):
        return self.q.Choice(title, value=value, checked=checked, disabled=disabled)


def no_terminal_error(missing: list[str], cause: Exception | None = None) -> SuiteError:
    detail = f" ({type(cause).__name__}: {cause})" if cause is not None else ""
    return SuiteError(
        "no interactive terminal detected (or prompts unavailable here); pass " + ", ".join(missing) + detail
    )


def ask_or_fail(missing: list[str], fn, *a, **k):
    """Shows one prompt; Ctrl-C/Esc cancels, and a prompt that can't be shown names the flags to pass instead."""
    try:
        value = fn(*a, **k)
    except Exception as e:
        raise no_terminal_error(missing, e) from e
    # questionary returns None when the user presses Ctrl-C or Esc.
    if value is None:
        raise Cancelled()
    return value


def resolve_inputs(args, interactive: bool, ask_factory=QuestionaryAsk) -> Inputs:
    flags_given = args.modalities is not None or args.languages is not None
    if args.capabilities and flags_given:
        raise SuiteError("give either --capabilities or --modalities/--languages, not both")

    missing = []
    if not args.capabilities:
        if args.modalities is None:
            missing.append("--modalities")
        if args.languages is None:
            missing.append("--languages")
    if not args.model_id:
        missing.append("--model-id")

    ask = None
    if missing:
        if not interactive:
            raise no_terminal_error(missing)
        try:
            ask = ask_factory()
        except Exception as e:
            raise no_terminal_error(missing, e) from e

    def prompt(fn, *a, **k):
        return ask_or_fail(missing, fn, *a, **k)

    if args.capabilities:
        capabilities = load_capabilities(args.capabilities)
    else:
        modalities = args.modalities
        if modalities is None:
            # Text is required, so the only real choice is whether image is also declared.
            accepts_images = prompt(ask.confirm, "Does this model also accept images? (text is always included)", default=False)
            modalities = ["text", "image"] if accepts_images else ["text"]
        languages = args.languages
        if languages is None:
            # Unticked to start: the cursor colour only shows on unticked rows, and it forces a deliberate pick.
            choices = [ask.choice(LANGUAGE_NAMES[code], code) for code in sorted(ALLOWED_LANGUAGES)]
            languages = prompt(
                ask.checkbox,
                "Which languages does it support?",
                choices,
                validate=lambda picked: bool(picked) or "Pick at least one language",
            )
        capabilities = parse_capabilities({"declared_modalities": list(modalities), "declared_languages": list(languages)})

    model_id = args.model_id or prompt(ask.text, "Model id (a stable name for this record):", default=suggest_model_id(args.model))
    if not model_id.strip():
        raise SuiteError("model id must not be empty")

    thresholds_path = Path(args.thresholds).resolve() if args.thresholds else DEFAULT_THRESHOLDS
    source = "interactive" if ask is not None else ("file" if args.capabilities else "flags")
    return Inputs(capabilities, model_id.strip(), thresholds_path, source)
