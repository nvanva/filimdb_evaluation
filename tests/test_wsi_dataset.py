import sys
from pathlib import Path
FILIMDB_PATH = Path(__file__).resolve().parent.parent
sys.path.append(str(FILIMDB_PATH))

from wsi.score import (
    BTSRNC,
    load_bts_rnc_dataset,
    load_russe_labels,
)


def test_load_bts_rnc_dataset():
    part2data, language = load_bts_rnc_dataset(BTSRNC)
    assert "train" in part2data, f"Train part was not loaded"
    assert "test" in part2data, f"Test part was not loaded"

    def _check_types(_context_idxs, _target_words, _contexts, _target_positions):
        assert all(isinstance(idx, int) for idx in _context_idxs)
        assert all(isinstance(word, str) for word in _target_words)
        assert all(isinstance(ctx, str) for ctx in _contexts)
        assert all(
            isinstance(pos, str) and len(pos.split(",")) == 1
            for pos in _target_positions
        )

    context_idxs, target_words, contexts, target_positions = part2data["train"]
    assert 3491 == len(target_words) == len(contexts) == len(target_positions) == len(context_idxs)
    _check_types(context_idxs, target_words, contexts, target_positions)

    context_idxs, target_words, contexts, target_positions = part2data["test"]
    assert 6556 == len(target_words) == len(contexts) == len(target_positions) == len(context_idxs)
    _check_types(context_idxs, target_words, contexts, target_positions)


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
