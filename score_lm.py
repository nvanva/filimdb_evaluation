from time import time
import os
import collections
import codecs
import random
import numpy as np

def _read_words(filename):
    with open(filename, "r") as f:
        return f.read().replace("\n", "<eos>").split()

def _build_vocab(filename):
    data = _read_words(filename)

    counter = collections.Counter(data)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

    words, _ = list(zip(*count_pairs))
    word_to_id = dict(zip(words, range(len(words))))
    id_to_word = {v: k for k, v in word_to_id.items()}

    return word_to_id, id_to_word

def _file_to_word_ids(filename, word_to_id):
    data = _read_words(filename)
    return [word_to_id[word] for word in data if word in word_to_id]

def load_dataset(data_path=None):

    train_path = os.path.join(data_path, "ptb.train.txt")
    dev_path = os.path.join(data_path, "ptb.valid.txt")
    test_path = os.path.join(data_path, "ptb.test.txt")

    word_to_id, id_to_word = _build_vocab(train_path)
    train_data = _file_to_word_ids(train_path, word_to_id)
    dev_data = _file_to_word_ids(dev_path, word_to_id)
    test_data = _file_to_word_ids(test_path, word_to_id)
    vocabulary = len(word_to_id)
    
    return train_data, dev_data, test_data, word_to_id, id_to_word

def save_preds(preds, preds_fname):
    """
    Save classifier predictions in format appropriate for scoring.
    """
    with codecs.open(preds_fname, 'w') as outp:
        for prob, pred_word, true_word in preds:
            print(true_word, prob, pred_word, sep='\t', file=outp)
    print('Predictions saved to %s' % preds_fname)


def load_preds(preds_fname):
    """
    Save classifier predictions in format appropriate for scoring.
    """
    with codecs.open(preds_fname,'r') as inp:
        pairs = [l.strip().split('\t') for l in inp.readlines()]
    true_words, probs, pred_words = zip(*pairs)
    probs = np.array(probs, dtype=np.float32)

    return probs, pred_words, true_words

def compute_perplexity(probs):
    perplexity = np.exp(-np.log(probs).sum() / len(probs))

    return perplexity

def score_preds(preds_path, ptb_path):

    probs, _, true_words = load_preds(preds_path)

    text = ' '.join(true_words).replace('<eos>', '\n').strip()
    
    test_path = os.path.join(ptb_path, 'ptb.test.txt')
    with open(test_path, 'r') as f:
        true_text = f.read().strip()

    # Check text is PTB
    if text != true_text[:len(text)]:
        raise ValueError('Received text does not match PTB text')

    # Perplexity calculation
    perplexity = compute_perplexity(probs)

    print('Perplexity:', perplexity)

    return perplexity
