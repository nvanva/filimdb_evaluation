from typing import List
import os
import codecs
import random
import numpy as np
from pathlib import Path
from pandas import read_csv

random.seed(42)
SCORED_PARTS = ('train', 'dev', 'train_small', 'dev_small', 'test')
TRANSLIT_PATH = Path(__file__).with_name("TRANSLIT")


def load_dataset(data_dir_path=None, parts: List[str] = SCORED_PARTS):
    part2ixy = {}
    for part in parts:
        path = os.path.join(data_dir_path, f'{part}.tsv')
        with open(path, 'r', encoding='utf-8') as rf:
            # first line is a header of the corresponding columns
            lines = rf.readlines()[1:]
            col_count = len(lines[0].strip('\n').split('\t'))
            if col_count == 2:
                strings, transliterations = zip(
                    *list(map(lambda l: l.strip('\n').split('\t'), lines))
                )
            elif col_count == 1:
                strings = list(map(lambda l: l.strip('\n'), lines))
                transliterations = None
            else:
                raise ValueError("wrong amount of columns")
        part2ixy[part] = (
            [f'{part}/{i}' for i in range(len(strings))],
            strings, transliterations,
        )
    return part2ixy


def load_transliterations_only(data_dir_path=None, parts: List[str] = SCORED_PARTS):
    part2iy = {}
    for part in parts:
        path = os.path.join(data_dir_path, f'{part}.tsv')
        with open(path, 'r', encoding='utf-8') as rf:
            # first line is a header of the corresponding columns
            lines = rf.readlines()[1:]
            col_count = len(lines[0].strip('\n').split('\t'))
            n_lines = len(lines)
            if col_count == 2:
                transliterations = [l.strip('\n').split('\t')[1] for l in lines]
            elif col_count == 1:
                transliterations = None
            else:
                raise ValueError("Wrong amount of columns")
        part2iy[part] = (
            [f'{part}/{i}' for i in range(n_lines)],
            transliterations,
        )
    return part2iy


def save_preds(preds, preds_fname):
    """
    Save classifier predictions in format appropriate for scoring.
    """
    with codecs.open(preds_fname, 'w') as outp:
        for idx, preds in preds:
            print(idx, *preds, sep='\t', file=outp)
    print('Predictions saved to %s' % preds_fname)


def load_preds(preds_fname, top_k=1):
    """
    Load classifier predictions in format appropriate for scoring.
    """
    kwargs = {
        "filepath_or_buffer": preds_fname,
        "names": ["id", "pred"],
        "sep": '\t',
    }

    pred_ids = read_csv(**kwargs, usecols=["id"])["id"].to_list()

    pred_y = {
        pred_id: y
        for pred_id, y in zip(
            pred_ids, read_csv(**kwargs, usecols=["pred"])["pred"]
        )
    }

    return pred_ids, pred_y


def compute_hit_k(preds, k=10):
    raise NotImplementedError


def compute_mrr(preds):
    raise NotImplementedError


def compute_acc_1(preds, true):
    right_answers = 0
    for pred, y in zip(preds, true):
        if pred[0] == y:
            right_answers += 1
    return right_answers / len(preds)


def score(preds, true):
    assert len(preds) == len(true), 'inconsistent amount of predictions and ground truth answers'
    acc_1 = compute_acc_1(preds, true)
    return {'acc@1': acc_1}


def score_preds(preds_path, data_dir, parts=SCORED_PARTS):
    part2iy = load_transliterations_only(data_dir, parts=parts)
    pred_ids, pred_dict = load_preds(preds_path)
    # pred_dict = {i:y for i,y in zip(pred_ids, pred_y)}
    scores = {}
    for part, (true_ids, true_y) in part2iy.items():
        if true_y is None:
            print('no labels for %s set' % part)
            continue
        pred_y = [pred_dict[i] for i in true_ids]
        score_values = score(pred_y, true_y)
        acc_1 = score_values['acc@1']
        print('%s set accuracy@1: %.2f' % (part, acc_1))
        scores[part] = score_values 
    return scores
