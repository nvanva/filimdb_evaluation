import subprocess
from pathlib import Path
from typing import Dict, Tuple, List
from collections import OrderedDict
from nltk import word_tokenize
import pandas as pd


DATA_DIR = Path(__file__).resolve().with_name("WSI")
BTSRNC = DATA_DIR / "bts-rnc"
RU = "ru"


def extract_wsi_data_if_not_exists():
    if DATA_DIR.exists():
        return
    subprocess.run(["tar", "-xvf" "WSI.tar.gz"])


def load_bts_rnc_dataset(
    data_path: Path = BTSRNC,
    parts: Tuple[str] = ("train", "test"),
):
    extract_wsi_data_if_not_exists()
    part2data = OrderedDict()
    for part in parts:
        df = pd.read_csv(data_path / f"{part}.csv", sep="\t", encoding="utf-8")
        target_words, tokens_lists, target_idxs, context_idxs = [], [], [], []
        for _, row in df.iterrows():
            tokens = word_tokenize(row.context)
            l, r = (idx for idx in row.positions.split(",", 1)[0].split("-"))
            word = row.context[int(l): int(r) + 1]

            target_words.append(row.word)
            tokens_lists.append(tokens)
            target_idxs.append(
                next(i for i, t in enumerate(tokens) if word in t)
            )
            context_idxs.append(row.context_id)

        part2data[part] = (context_idxs, target_words, tokens_lists, target_idxs)

    return part2data, RU


def load_russe_labels(
    data_path: Path = BTSRNC,
    parts: Tuple[str] = ("train", "test"),
):
    extract_wsi_data_if_not_exists()
    part2labels = OrderedDict()
    for part in parts:
        df = pd.read_csv(
            data_path / f"{part}.csv", sep="\t", encoding="utf-8",
            dtype={"gold_sense_id": str},
            usecols=["context_id", "gold_sense_id", "word"],
        )
        context_idxs, labels, target_words = [], [], []
        for _, row in df.iterrows():
            context_idxs.append(row.context_id)
            labels.append(row.gold_sense_id)
            target_words.append(row.word)
        if part in ("test",):
            labels = None
        part2labels[part] = (context_idxs, labels, target_words)
    return part2labels


def load_dataset(
    dataset: str,
) -> Tuple[Dict[str, Tuple[List[int], List, List, List]], str]:
    if dataset == "bts-rnc":
        return load_bts_rnc_dataset()
    raise ValueError(f'Dataset "{dataset}" is not available')


def load_labels(
    dataset: str,
    data_path: Path = DATA_DIR,
    parts: Tuple[str] = ("train", "test"),
) -> Dict[str, Tuple[List[int], List[str], List[str]]]:
    if dataset == "bts-rnc":
        return load_russe_labels(data_path / dataset, parts=parts)
    raise ValueError(f'Dataset "{dataset}" is not available')


def save_preds(
    idx2label: Dict[int, str],
    dataset: str,
    preds_fname: Path,
):
    if dataset == "bts-rnc":
        part2labels = load_russe_labels(BTSRNC)
        context_idxs = [
            idx for part_idxs, _, _ in part2labels.values()
            for idx in part_idxs
        ]
        assert len(context_idxs) == len(idx2label), \
            f"Number of predicted instances " \
            f"doesn't match gold number of instances"
        labels_to_save = [idx2label[idx] for idx in context_idxs]
        df = pd.DataFrame.from_dict({
            "context_id": context_idxs,
            "predicted_label": labels_to_save
        })
        df.to_csv(preds_fname)
    else:
        raise ValueError(f'Dataset "{dataset}" is not available')
