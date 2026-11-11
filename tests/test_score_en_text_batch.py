import inspect

import score_en_text


def test_classifier_scores_one_prompt_per_batch_by_default():
    assert score_en_text.DEFAULT_BATCH_SIZE == 1
    default = inspect.signature(score_en_text.score_records).parameters["batch_size"].default
    assert default == score_en_text.DEFAULT_BATCH_SIZE
