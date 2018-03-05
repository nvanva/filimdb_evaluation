from time import time
import os
import codecs
import random

random.seed(3)  # set random seed for each run of the script to produce the same results
SCORED_PARTS = ('train', 'dev', 'test')
ALL_PARTS = (*SCORED_PARTS, 'train_unlabeled')

def load_dataset_fast(data_dir='FILIMDB', parts=SCORED_PARTS):
    """
    Loads data from specified directory. Returns dictionary part->(list of texts, list of corresponding labels).
    """
    part2xy = {}  # tuple(list of texts, list of their labels) for train and test parts
    for part in parts:
        print('Loading %s set ' % part)

        xpath = os.path.join(data_dir, '%s.texts' % part)
        with codecs.open(xpath, 'r', encoding='utf-8') as inp:
            texts = [s.strip() for s in inp.read().strip().split('\n')]

        ypath = os.path.join(data_dir, '%s.labels' % part)
        if os.path.exists(ypath):
            with codecs.open(ypath, 'r', encoding='utf-8') as inp:
                labels = [s.strip() for s in inp.readlines()]
            assert len(labels) == len(texts), 'Number of labels and texts differ in %s set!' % part
            for cls in set(labels):
                print(cls, sum((1 for l in labels if l == cls)))

        else:
            labels = None
            print('unlabeled', len(texts))

        part2xy[part] = (['%s/%d' % (part,i) for i in range(len(texts))], texts, labels)
    return part2xy


def load_dataset(data_dir='ILIMDB',parts=('train', 'dev', 'test', 'train_unlabeled')):
    """
    Loads data from specified directory. Returns dictionary part->(list of texts, list of corresponding labels).
    """
    part2xy = {} # tuple(list of texts, list of their labels) for train and test parts
    for part in parts:
        print('Loading %s set ' % part)

        unlabeled_subdir = os.path.join(data_dir, part, 'unlabeled')
        unlabeled = os.path.exists(unlabeled_subdir)
        examples = []

        if unlabeled:
            load_dir(unlabeled_subdir, None, examples)
        else:
            for cls in ('pos', 'neg'):
                subdir = os.path.join(data_dir, part, cls)
                load_dir(subdir, cls, examples)
        # shuffle examples: if the classifiers overfits to a particular order of labels,
        # it will show bad results on dev/test set;
        # train set should be shuffled by the train() function if the classifier can overfit to the order!
        if part != 'train':
            random.shuffle(examples)
        ids, texts, labels = list(zip(*examples))  # convert list of (text,label) pairs to 2 parallel lists
        part2xy[part] = (ids, texts, None) if unlabeled else (ids, texts, labels)
        for cls in set(labels):
            print(cls, sum((1 for l in labels if l==cls)))
    return part2xy


def load_dir(subdir, cls, examples):
    st = time()
    for fname in os.listdir(subdir):
        fpath = os.path.join(subdir, fname)
        with codecs.open(fpath, mode='r', encoding='utf-8') as inp:
            s = ' '.join(inp.readlines())  # Join all lines into single line
            examples.append((fpath, s, cls))
    print(subdir, time()-st)


def score(y_pred, y_true):
    assert len(y_pred)==len(y_true), 'Received %d but expected %d labels' % (len(y_pred), len(y_true))
    correct = sum(y1 == y2 for y1, y2 in zip(y_pred, y_true))
    print('Number of correct/incorrect predictions: %d/%d' % (correct, len(y_pred)))
    acc = 100.0 * correct / len(y_pred)
    return acc

def save_preds(preds, preds_fname):
    """
    Save classifier predictions in format appropriate for scoring.
    """
    with codecs.open(preds_fname, 'w') as outp:
        for a, b in preds:
            print(a, b, sep='\t', file=outp)
    print('Predictions saved to %s' % preds_fname)


def load_preds(preds_fname):
    """
    Save classifier predictions in format appropriate for scoring.
    """
    with codecs.open(preds_fname,'r') as inp:
        pairs = [l.strip().split('\t') for l in inp.readlines()]
    ids, preds = zip(*pairs)
    return ids, preds


def score_preds(preds_fname, data_dir='FILIMDB'):
    part2xy = load_dataset_fast(data_dir=data_dir, parts=SCORED_PARTS)
    return score_preds_loaded(part2xy, preds_fname)


def score_preds_loaded(part2xy, preds_fname):
    pred_ids, pred_y = load_preds(preds_fname)
    pred_dict = {i: y for i, y in zip(pred_ids, pred_y)}
    scores = {}
    for part, (true_ids, _, true_y) in part2xy.items():
        if true_y is None:
            print('no labels for %s set' % part)
            continue

        pred_y = [pred_dict[i] for i in true_ids]
        acc = score(pred_y, true_y)
        print('%s set accuracy: %.2f' % (part, acc))
        scores[part] = acc
    return scores
