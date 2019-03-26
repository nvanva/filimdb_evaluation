import argparse
import numpy as np
from tqdm import tqdm
from itertools import chain
from lm import train, next_proba_gen
from score_lm import load_dataset, save_preds, score_preds

PREDS_FNAME = 'preds.tsv'

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


def predict_probs(model, id_to_word, data, name, top_k=3, bs=100):
    data = np.array(data)
    tail = len(data)//bs*bs
    X, X_tail = data[0:tail].reshape(bs, -1).T, data[tail:-1].reshape(1, -1).T  # X[i+1,j] is the next word after X[i,j]
    assert (np.concatenate([X[:, i] for i in range(X.shape[1])] + [X_tail[:, 0]]) == data[:-1]).all()
    Y, Y_tail = data[1:tail+1].reshape(bs, -1).T, data[tail+1:].reshape(1, -1).T

    preds = []
    desc = 'Generate probability of the next word. Dataset: "{}", batch size: {}'.format(name, bs)
    X_protected_it = ProtectedTokenIterator(X)
    for cur_id, next_id, softmax in zip(tqdm(X, desc=desc), Y, next_proba_gen(X_protected_it, model)):
        X_protected_it.allow_next = True
        append_prediction(preds, cur_id, next_id, softmax, id_to_word, top_k, bs)

    r = map(lambda lol: np.array(lol).T.ravel(), zip(*preds))  # convert to triple of 1D-arrays, making next column continuation of the previous one

    bs = 1  # process tail with batch size 1
    preds = []
    desc = 'Generate probability of the next word. Dataset: "{}" (tail), batch size: {}'.format(name, bs)
    X_protected_it = ProtectedTokenIterator(X_tail)
    for cur_id, next_id, softmax in zip(tqdm(X_tail, desc=desc), Y_tail, next_proba_gen(X_protected_it, model)):
        X_protected_it.allow_next = True
        append_prediction(preds, cur_id, next_id, softmax, id_to_word, top_k, bs)

    rtail = map(lambda lol: np.array(lol).T.ravel(), zip(*preds))
    return chain(zip(*r), zip(*rtail))  # convert to iterable over triples


def append_prediction(preds, cur_id, next_id, softmax, id_to_word, top_k, bs):
    assert softmax.shape == (bs, len(id_to_word),)
    check_softmax(softmax)

    cur_word = [id_to_word[i] for i in cur_id]
    true_prob = softmax[np.arange(bs), next_id]
    true_rank = (softmax >= true_prob.reshape(-1, 1)).sum(axis=-1) - 1

    top_k_idxs = np.argpartition(softmax, -top_k, axis=-1)[:, -top_k:]
    allrows_matr = np.arange(bs)[:, np.newaxis].repeat(top_k, axis=1)
    top_k_idxs_order = softmax[allrows_matr, top_k_idxs].argsort(axis=-1)[:, ::-1]
    top_k_idxs_sorted = top_k_idxs[allrows_matr, top_k_idxs_order]

    top_k_words = [[id_to_word[idx] for idx in top_k_idxs_sorted[:, i]] for i in range(top_k)]
    # top_k_words = []
    preds.append([cur_word] + top_k_words + [true_prob, true_rank])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ptb_path', default='PTB', help='Path to PTB data')
    args = parser.parse_args()

    raw_data = load_dataset(args.ptb_path)
    train_data, dev_data, test_data, word_to_id, id_to_word = raw_data

    model = train(train_data, word_to_id, id_to_word)

    allpreds = [['prev', 'pred1', 'pred2', 'pred3', 'true_prob', 'true_rank']]
    allpreds.extend(predict_probs(model, id_to_word, train_data, 'train'))
    allpreds.extend(predict_probs(model, id_to_word, dev_data, 'valid'))
    allpreds.extend(predict_probs(model, id_to_word, test_data, 'test'))

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