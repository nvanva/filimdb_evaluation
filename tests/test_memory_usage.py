import re
import io
from contextlib import redirect_stdout
from memory_profiler import profile
from pathlib import Path
import subprocess

import score
import score_lm
import score_translit

from score import score_preds, FILIMDB_PATH
from score_lm import score_preds as score_preds_lm, PTB_PATH
from score_translit import score_preds as score_preds_translit, TRANSLIT_PATH

from evaluate_lm import PREDS_FNAME as LM_PREDS_FNAME

score_preds = profile(score_preds)
score.load_labels_only = profile(score.load_labels_only)
score.score_preds_loaded = profile(score.score_preds_loaded)
score.load_preds = profile(score.load_preds)

score_preds_lm = profile(score_preds_lm)
score_lm.load_preds = profile(score_lm.load_preds)

score_preds_translit = profile(score_preds_translit)
score_translit.load_preds = profile(score_translit.load_preds)
score_translit.load_dataset = profile(score_translit.load_dataset)
score_translit.score = profile(score_translit.score)


def profile_function_memory(function, *args, **kwargs):
    """
    Calls "function" and tracks how much memory is allocated after each line of "function" code
    :param function: memory usage of this function is tracked
    :param args: "function" arguments
    :param kwargs: "function" arguments
    :return: "function" output value and maximum memory used by the "function"
    """
    profiled = io.StringIO()
    with redirect_stdout(profiled):
        returned_value = function(*args, **kwargs)

    used_memory_by_line = []
    print(profiled.getvalue())
    for line in profiled.getvalue().split('\n')[4:]:
        if "MiB" in line:
            # extracting memory used after each line of "function" code
            used_memory, _ = re.findall(r"(\d+\.\d+) MiB", line)
            used_memory_by_line.append(float(used_memory))
    return returned_value, max(used_memory_by_line) - min(used_memory_by_line)


def test_score_memory_usage():
    scoring_result, used_memory = profile_function_memory(
        score_preds,
        FILIMDB_PATH.parent / "preds.tsv",
        FILIMDB_PATH,
    )
    message = f"Memory used by IMDB evaluation: {used_memory}"
    print(message)
    assert used_memory < 30, message


def test_score_lm_memory_usage():
    if not LM_PREDS_FNAME.exists():
        subprocess.run([
            "python",
            Path(__file__).parent.parent / "evaluate_lm.py",
            "evaluate",
        ])

    scoring_result, used_memory = profile_function_memory(
        score_preds_lm,
        LM_PREDS_FNAME,
        PTB_PATH,
    )
    message = f"Memory used by Language Model evaluation: {used_memory}"
    print(message)
    assert used_memory < 65, message


def test_score_translit_memory_usage():
    if not TRANSLIT_PATH.with_name("preds_translit.tsv").exists():
        subprocess.run([
            "python",
            Path(__file__).parent.parent / "evaluate_translit.py",
        ])

    scoring_result, used_memory = profile_function_memory(
        score_preds_translit,
        TRANSLIT_PATH.with_name("preds_translit.tsv"),
        TRANSLIT_PATH,
    )
    message = f"Memory used by Transliteration evaluation: {used_memory}"
    print(message)
    assert used_memory < 75, message
