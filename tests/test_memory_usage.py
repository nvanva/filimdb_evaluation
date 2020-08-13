import re
import io
from contextlib import redirect_stdout
from memory_profiler import profile
from pathlib import Path
import subprocess

from score import score_preds, FILIMDB_PATH
from score_lm import score_preds as score_preds_lm, PTB_PATH
from score_translit import score_preds as score_preds_translit, TRANSLIT_PATH


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
    for line in profiled.getvalue().split('\n')[4:]:
        if "MiB" in line:
            # extracting memory used after each line of "function" code
            used_memory, _ = re.findall(r"(\d+\.\d+) MiB", line)
            used_memory_by_line.append(float(used_memory))
    return returned_value, max(used_memory_by_line) - used_memory_by_line[0]


@profile
def score_memory_profiler(*args, **kwargs):
    return score_preds(*args, **kwargs)


def test_score_memory_usage():
    scoring_result, used_memory = profile_function_memory(
        score_memory_profiler,
        FILIMDB_PATH.parent / "preds.tsv",
        FILIMDB_PATH,
    )
    print(f"Memory used by IMDB evaluation: {used_memory}")
    assert used_memory < 100


@profile
def score_lm_memory_profiler(*args, **kwargs):
    return score_preds_lm(*args, **kwargs)


def test_score_lm_memory_usage():
    if not PTB_PATH.with_name("preds_lm.tsv").exists():
        subprocess.run([
            "python",
            Path(__file__).parent.parent / "evaluate_lm.py",
            "evaluate",
        ])

    scoring_result, used_memory = profile_function_memory(
        score_lm_memory_profiler,
        PTB_PATH.with_name("preds_lm.tsv"),
        PTB_PATH,
    )
    print(f"Memory used by Language Model evaluation: {used_memory}")
    assert used_memory < 100


@profile
def score_translit_memory_profiler(*args, **kwargs):
    return score_preds_translit(*args, **kwargs)


def test_score_translit_memory_usage():
    if not TRANSLIT_PATH.with_name("preds_translit.tsv").exists():
        subprocess.run([
            "python",
            Path(__file__).parent.parent / "evaluate_translit.py",
        ])

    scoring_result, used_memory = profile_function_memory(
        score_translit_memory_profiler,
        TRANSLIT_PATH.with_name("preds_translit.tsv"),
        TRANSLIT_PATH,
    )
    print(f"Memory used by Transliteration evaluation: {used_memory}")
    assert used_memory < 100
