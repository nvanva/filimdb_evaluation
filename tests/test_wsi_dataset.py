from wsi.dataset import (
    BTSRNC,
    load_bts_rnc_dataset,
    load_russe_labels,
)


def test_load_bts_rnc_dataset():
    part2data, language = load_bts_rnc_dataset(BTSRNC)
    assert "train" in part2data, f"Train part was not loaded"
    assert "test" in part2data, f"Test part was not loaded"
    context_idxs, target_words, tokens_lists, target_idxs = part2data["train"]
    assert len(target_words) == 3491
    assert len(target_words) == len(tokens_lists)
    assert len(tokens_lists) == len(target_idxs)
    assert len(target_idxs) == len(context_idxs)

    context_idxs, target_words, tokens_lists, target_idxs = part2data["test"]
    assert len(target_words) == 6556
    assert len(target_words) == len(tokens_lists)
    assert len(tokens_lists) == len(target_idxs)
    assert len(target_idxs) == len(context_idxs)


def test_load_russe_labels():
    part2data = load_russe_labels(BTSRNC)
    assert "train" in part2data, f"Train part was not loaded"
    assert "test" in part2data, f"Test part was not loaded"
    context_idxs, labels, target_words = part2data["train"]
    assert 3491 == len(context_idxs) == len(labels) == len(target_words)
    assert all(isinstance(idx, int) for idx in context_idxs)
    assert all(isinstance(label, str) for label in labels)
    assert all(isinstance(tw, str) for tw in target_words)

    context_idxs, labels, target_words = part2data["test"]
    assert 6556 == len(context_idxs) == len(target_words)
    assert labels is None
    assert all(isinstance(idx, int) for idx in context_idxs)
    assert all(isinstance(tw, str) for tw in target_words)
