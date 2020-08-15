import os
import collections
import codecs
import numpy as np
from pandas import read_csv
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

    df = read_csv(
        preds_fname, sep='\t', float_precision="high",
        usecols=["prev", "true_prob", "true_rank", "kl_uniform", "kl_unigram"]
    )
    prevs = df["prev"].to_list()
    del df["prev"]
    true_probs = np.float32(df["true_prob"])
    del df["true_prob"]
    true_ranks = np.int32(df["true_rank"])
    del df["true_rank"]
    kl_uniform = np.float32(df["kl_uniform"])
    del df["kl_uniform"]
    kl_unigram = np.float32(df["kl_unigram"])
    del df["kl_unigram"]

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

    with open(os.path.join(ptb_path, "ptb.train.txt"), "r") as f:
        train_text = f.read().strip().replace("\n", "<eos>")

    with open(os.path.join(ptb_path, "ptb.valid.txt"), "r") as f:
        dev_text = f.read().strip().replace("\n", "<eos>")

    with open(os.path.join(ptb_path, "ptb.test.txt"), "r") as f:
        test_text = f.read().strip().replace("\n", "<eos>")

    ptb_dataset = [
        ('train', train_text, train_text.count(" ") + 1),
        ('valid', dev_text, dev_text.count(" ") + 1),
        ('test', test_text, test_text.count(" ") + 1),
    ]

    scores = dict()
    for name, text, len_text in ptb_dataset:
        # Check text is PTB
        if ' '.join(recieved_text[:len_text]) != text:
            raise Exception(f'Received text does not match PTB text')

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
