import sys
from pathlib import Path

# Scripts import each other by bare module name, so tests need scripts/ on the path the same way.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
