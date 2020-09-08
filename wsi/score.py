from pathlib import Path
from typing import List, Dict, Tuple
from itertools import groupby
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from wsi.dataset import load_labels, DATA_DIR


def score_part(
    gold_labels: List[str],
    pred_labels: List[str],
    target_words: List[str]
) -> Dict[str, float]:
    """
    Computes weighted average of the wsi metrics.
    :param gold_labels: true labels for the dataset
    :param pred_labels: predicted labels for the dataset
    :param target_words: ambiguous words that are used to group sentences
    :return: dict of scores
    """
    instances = sorted(
        zip(target_words, range(len(target_words))), key=lambda it: it[0],
    )
    weighted_avg = 0.0
    for target_word, grouped_instances in groupby(instances, lambda it: it[0]):
        idxs = [idx for _, idx in grouped_instances]
        grouped_gold = [gold_labels[idx] for idx in idxs]
        grouped_pred = [pred_labels[idx] for idx in idxs]
        ari = adjusted_rand_score(grouped_gold, grouped_pred)
        weighted_avg += len(grouped_gold) * ari
    return {"ARI": round(weighted_avg / len(target_words), 6)}


def score_preds(
    dataset: str,
    preds_fname: Path,
    data_path: Path = DATA_DIR,
    parts: Tuple[str] = ("train", "test"),
) -> Dict[str, Dict[str, float]]:
    """
    Scores predicted labels from the "preds_fname" file.
    :param dataset: dataset whose labels will be compared
        with the labels from the "preds_fname" file.
    :param preds_fname: file that contains predicted labels
    :return: dict of scores for each part of the dataset
    """
    df = pd.read_csv(preds_fname)
    idx2label = {r.context_id: r.predicted_label for _, r in df.iterrows()}
    part2labels = load_labels(dataset, data_path=data_path, parts=parts)
    part2scores = dict()
    for part, data in part2labels.items():
        context_idxs, gold_labels, target_words = data
        if gold_labels is None:
            continue
        pred_labels = [idx2label[idx] for idx in context_idxs]
        part2scores[part] = score_part(gold_labels, pred_labels, target_words)
    return part2scores
