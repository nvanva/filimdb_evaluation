import os
import collections
import codecs
import numpy as np
from pathlib import Path

PTB_PATH = Path(__file__).with_name("PTB")


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

    return train_data, dev_data, test_data, word_to_id, id_to_word


def save_preds(preds, preds_fname):
    """
    Save classifier predictions in format appropriate for scoring.
    """
    with codecs.open(preds_fname, 'w') as outp:
        for vals in preds:
            print(*vals, sep='\t', file=outp)
    print('Predictions saved to %s' % preds_fname)


def load_preds(preds_fname):
    """
    Load classifier predictions in format appropriate for scoring.
    """
    prevs = []
    with codecs.open(preds_fname, 'r') as inp:
        for l in inp.readlines()[1:]:
            prevs.append(l.strip().split('\t', 1)[0])

    true_probs = np.zeros(len(prevs), dtype=np.float32)
    true_ranks = np.zeros(len(prevs), dtype=np.int32)
    kl_uniform = np.zeros(len(prevs), dtype=np.float32)
    kl_unigram = np.zeros(len(prevs), dtype=np.float32)
    with codecs.open(preds_fname, 'r') as inp:
        for i, l in enumerate(inp.readlines()[1:]):
            true_prob, true_rank, kl_uniform_, kl_unigram_ = l.strip().rsplit('\t', 4)[-4:]
            true_probs[i] = np.float32(true_prob)
            true_ranks[i] = np.int32(true_rank)
            kl_uniform[i] = np.float32(kl_uniform_)
            kl_unigram[i] = np.float32(kl_unigram_)

    return prevs, true_probs, true_ranks, kl_uniform, kl_unigram


def compute_perplexity(probs):
    return np.exp(-np.log(probs).sum() / len(probs))


def compute_hit_k(ranks, k=10):
    mask = np.where(ranks < k)[0]
    return float(len(mask)) / len(ranks)


def compute_average_rank(ranks):
    return np.mean(ranks)


def compute_average_kl(kl_divergence):
    return np.mean(kl_divergence)


def score_preds(preds_path, ptb_path=PTB_PATH):
    data = load_preds(preds_path)
    recieved_text, probs, ranks, kl_uniform, kl_unigram = data

    train_path = os.path.join(ptb_path, "ptb.train.txt")
    dev_path = os.path.join(ptb_path, "ptb.valid.txt")
    test_path = os.path.join(ptb_path, "ptb.test.txt")

    train_text = _read_words(train_path)[:-1]
    dev_text = _read_words(dev_path)[:-1]
    test_text = _read_words(test_path)[:-1]

    ptb_dataset = [
        ('train', train_text),
        ('valid', dev_text),
        ('test', test_text),
    ]

    scores = dict()
    for name, text in ptb_dataset:
        len_text = len(text)

        # Check text is PTB
        if ' '.join(recieved_text[:len_text]) != ' '.join(text):
            raise Exception('Received text does not match PTB text')

        # Perplexity calculation
        perplexity = compute_perplexity(probs[:len_text])  
        hit_k = compute_hit_k(ranks[:len_text])
        avg_rank = compute_average_rank(ranks[:len_text])
        avg_kl_uniform = compute_average_kl(kl_uniform[:len_text])
        avg_kl_unigram = compute_average_kl(kl_unigram[:len_text])

        scores[name] = {
            'perplexity': perplexity,
            'hit@10': hit_k,
            'avg_rank': avg_rank,
            'avg_kl_uniform': avg_kl_uniform,
            'avg_kl_unigram': avg_kl_unigram,
        }

        probs = probs[len_text:]
        ranks = ranks[len_text:]
        kl_uniform = kl_uniform[len_text:]
        kl_unigram = kl_unigram[len_text:]
        recieved_text = recieved_text[len_text:]

    return scores
