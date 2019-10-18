import argparse
import numpy as np
from tqdm import tqdm
from itertools import chain
from collections import Counter

from lm import train, next_proba_gen
from score_lm import load_dataset, save_preds, score_preds
import datetime

PREDS_FNAME = 'preds.tsv'

def normalize(x):
    return x / x.sum(axis=-1)


def train_unigram_model(token_list, word_to_id):
    vocab_size = len(word_to_id)
    counter = Counter(token_list)
    unigram_probs = normalize(np.array([counter[i] for i in range(vocab_size)]))
    return unigram_probs


def check_softmax(softmax):
    if (softmax < 0).any():
        print("Some probabilities are <0")

    if (softmax > 1).any():
        print("Some probabilities are >1")

    eps = 1e-3
    prob = softmax.sum(axis=-1)

    dontsum = abs(1.0 - prob) > eps
    if dontsum.any():
        raise Exception("Sum of the probabilities isn't equal to 1. Sum: {}".format(prob[dontsum]))


def sampling(model, unigram_probs, word_to_id, id_to_word, size, tokens=None, temperature=1.0):
    if tokens is None:
        idx = np.random.choice(len(id_to_word), p=unigram_probs)
        tokens = [id_to_word[idx]]

    hidden = None
    eos_id = word_to_id['<eos>']
    sequence = [word_to_id[word] for word in tokens]
    for _ in range(size):
        softmax, hidden = next(next_proba_gen([[sequence[-1]]], model, hidden=hidden))
        softmax = softmax[0]
        if temperature != 1.0:
            softmax = np.float_power(softmax, 1.0/temperature)
            softmax /= softmax.sum()
        idx = np.random.choice(list(range(len(softmax))), 1, p=softmax)
        sequence.append(idx[0])

    gen_text = [id_to_word[idx] for idx in sequence]
    return ' '.join(gen_text)


class ProtectedTokenIterator(object):
    """
    ProtectedTokenIterator doesn't allow LM to look forward (get actual next token before returning predicted
    distribution for it.
    """
    def __init__(self, tokens):
        self.it = iter(tokens)
        self.allow_next = True

    def __iter__(self):
        return self

    def __next__(self):
        if not self.allow_next:
            raise Exception('Next token requested before result for previous token was returned.')
        self.allow_next = False
        return next(self.it)


def predict_probs(model, id_to_word, data, unigram_probs, name, top_k=3, bs=10000):
    data = np.array(data)
    tail = len(data)//bs*bs
    X, X_tail = data[0:tail].reshape(bs, -1).T, data[tail:-1].reshape(1, -1).T  # X[i+1,j] is the next word after X[i,j]
    assert (np.concatenate([X[:, i] for i in range(X.shape[1])] + [X_tail[:, 0]]) == data[:-1]).all()
    Y, Y_tail = data[1:tail+1].reshape(bs, -1).T, data[tail+1:].reshape(1, -1).T

    preds = []
    desc = 'Generate probability of the next word. Dataset: "{}", batch size: {}'.format(name, bs)
    X_protected_it = ProtectedTokenIterator(X)
    for cur_id, next_id, (softmax, hidden_state) in zip(tqdm(X, desc=desc), Y, next_proba_gen(X_protected_it, model)):
        X_protected_it.allow_next = True
        append_prediction(preds, cur_id, next_id, softmax, unigram_probs, id_to_word, top_k, bs)

    r = map(lambda lol: np.array(lol).T.ravel(), zip(*preds))  # convert to triple of 1D-arrays, making next column continuation of the previous one

    bs = 1  # process tail with batch size 1
    preds = []
    desc = 'Generate probability of the next word. Dataset: "{}" (tail), batch size: {}'.format(name, bs)
    X_protected_it = ProtectedTokenIterator(X_tail)
    for cur_id, next_id, (softmax, hidden_state) in zip(tqdm(X_tail, desc=desc), Y_tail, next_proba_gen(X_protected_it, model)):
        X_protected_it.allow_next = True
        append_prediction(preds, cur_id, next_id, softmax, unigram_probs, id_to_word, top_k, bs)

    rtail = map(lambda lol: np.array(lol).T.ravel(), zip(*preds))
    return chain(zip(*r), zip(*rtail))  # convert to iterable over triples


def compute_kl_divergence(P, Q, is_uniform=False):
    eps = 1e-5
    P_mod = P + eps
    if is_uniform:
        n = P_mod.shape[1]
        divergence = np.log(n)+np.sum(P_mod*np.log(P_mod), axis=1)
    else:
        Q_mod = Q + eps
        divergence = np.sum(P_mod*np.log(P_mod/Q_mod), axis=1)

    return divergence


def append_prediction(preds, cur_id, next_id, softmax, unigram_probs, id_to_word, top_k, bs):
    assert softmax.shape == (bs, len(id_to_word),)
    check_softmax(softmax)

    cur_word = [id_to_word[i] for i in cur_id]
    true_prob = softmax[np.arange(bs), next_id]
    true_rank = (softmax >= true_prob.reshape(-1, 1)).sum(axis=-1) - 1

    kl_uniform = compute_kl_divergence(softmax, None, is_uniform=True)
    kl_unigram = compute_kl_divergence(softmax, unigram_probs)
    kl_divergences = [kl_uniform, kl_unigram]

    top_k_idxs = np.argpartition(softmax, -top_k, axis=-1)[:, -top_k:]
    allrows_matr = np.arange(bs)[:, np.newaxis].repeat(top_k, axis=1)
    top_k_idxs_order = softmax[allrows_matr, top_k_idxs].argsort(axis=-1)[:, ::-1]
    top_k_idxs_sorted = top_k_idxs[allrows_matr, top_k_idxs_order]

    top_k_words = [[id_to_word[idx] for idx in top_k_idxs_sorted[:, i]] for i in range(top_k)]
    preds.append([cur_word] + top_k_words + [true_prob, true_rank] + kl_divergences)


def datetime_str():
    return datetime.datetime.now().ctime()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ptb_path', default='PTB', help='Path to PTB data')
    args = parser.parse_args()

    raw_data = load_dataset(args.ptb_path)
    train_data, dev_data, test_data, word_to_id, id_to_word = raw_data

    print(datetime_str(), 'Training model ...')
    model = train(train_data, word_to_id, id_to_word)
    print(datetime_str(), 'Training model finished.')
    unigram_probs = train_unigram_model(train_data, word_to_id)

    allpreds = [['prev', 'pred1', 'pred2', 'pred3', 
                 'true_prob', 'true_rank', 'kl_uniform', 'kl_unigram']]

    print(datetime_str(), 'Testing model ...')
    allpreds.extend(predict_probs(model, id_to_word, train_data, unigram_probs, 'train'))
    allpreds.extend(predict_probs(model, id_to_word, dev_data, unigram_probs, 'valid'))
    allpreds.extend(predict_probs(model, id_to_word, test_data, unigram_probs, 'test'))
    print(datetime_str(), 'Testing model finished.')

    save_preds(allpreds, preds_fname=PREDS_FNAME)

    scores = score_preds(PREDS_FNAME, args.ptb_path)

    for method_name, method_scores in scores.items():
        for metric, value in method_scores.items():
            print('{} {}: {}'.format(method_name.capitalize(), metric, value))
        print()

    # comments = []
    # for score in ['perplexity', 'hit@10', 'avg_rank']:
    #     for part in ['train', 'valid']:
    #         comments.append('{}_{}: {:.3f}'.format(part, score, scores[part][score]))
    # comment = ', '.join(comments)
    # print(comment)

if __name__=='__main__':
    main()