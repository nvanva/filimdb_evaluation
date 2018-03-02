import random
from time import time
from classifier import train, classify  # classifier.py should be in the same directory
import os
import codecs
import random

random.seed(3)  # set random seed for each run of the script to produce the same results

def load_dataset_fast(data_dir='FILIMDB', parts=('train','dev','test')):
    """
    Loads data from specified directory. Returns dictionary part->(list of texts, list of corresponding labels).
    """
    part2xy = {}  # tuple(list of texts, list of their labels) for train and test parts
    for part in parts:
        print('Loading %s set ' % part)

        xpath = os.path.join(data_dir, '%s.texts' % part)
        with codecs.open(xpath, 'r', encoding='utf-8') as inp:
            texts = [s.strip() for s in inp.read().split('\n')]

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
    print('Accuracy: %.2f' % acc)
    return acc

def preds_fname(part):
    return '%s_preds.tsv' % part

def save_preds(ids, preds, part):
    """
    Save classifier predictions in format appropriate for scoring.
    """
    assert len(ids)==len(preds)
    fname = preds_fname(part)
    with codecs.open(fname, 'w') as outp:
        for a, b in zip(ids, preds):
            print(a, b, sep='\t', file=outp)
    print('Predictions saved to %s' % fname)


def load_preds(part, dir='.'):
    """
    Save classifier predictions in format appropriate for scoring.
    """
    fname = os.path.join(dir, preds_fname(part))
    with open(fname,'r') as inp:
        pairs = [l.strip().split('\t') for l in inp.readlines()]
    ids, preds = zip(*pairs)
    return ids, preds


def score_preds(part, preds_dir='.'):
    pred_ids, pred_y = load_preds(part, dir=preds_dir)
    true_ids, _, true_y = load_dataset_fast(parts=[part])[part]
    if true_y is None:
        print('no labels for %s set' % part)
        return

    pred_dict = {i: y for i, y in zip(pred_ids, pred_y)}
    pred_y = [pred_dict[i] for i in true_ids]
    score(pred_y, true_y)


def check_preds():
    for part in ('dev', 'test'):
        score_preds(part)


def main():
    part2xy = load_dataset_fast('FILIMDB')
    train_ids, train_texts, train_labels = part2xy['train']

    print('\nTraining classifier on %d examples from train set ...' % len(train_texts))
    st = time()
    params = train(train_texts, train_labels)
    print('Classifier trained in %.2fs' % (time()-st))

    for part, (ids, x, y) in part2xy.items():
        print('\nClassifying %s set with %d examples ...' % (part, len(x)))
        st = time()
        preds = classify(x, params)
        print('%s set classified in %.2fs' % (part, time() - st))
        save_preds(ids, preds, part)

        if y is None:
            print('no labels for %s set' % part)
        else:
            score(preds, y)

    print('\nChecking saved predictions ...')
    check_preds()

if __name__=='__main__':
    main()
