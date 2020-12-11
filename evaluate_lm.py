import fire
import numpy as np
from tqdm import tqdm
from itertools import chain
from collections import Counter
from pathlib import Path

from lm import train, next_proba_gen
from score_lm import load_dataset, save_preds, score_preds, PTB_PATH
import datetime

PREDS_FNAME = Path(__file__).with_name("preds_lm.tsv")


def datetime_str():
    return datetime.datetime.now().ctime()


class ProtectedTokenIterator(object):
    """
    ProtectedTokenIterator doesn't allow LM to look forward 
    (get actual next token before returning predicted distribution for it.
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


def normalize(x):
    return x / x.sum(axis=-1)


def ancestral_sampling_batch(model, 
                             unigram_probs, 
                             word_to_id, 
                             id_to_word, 
                             size, 
                             batch_size=20,
                             start_text=None, 
                             temperature=1.0):

    unk_id = word_to_id['<unk>']
    if start_text is None:
        idxs = np.random.choice(len(id_to_word), size=batch_size, p=unigram_probs)
        tokens = idxs.reshape((batch_size, 1))
    else:
        idxs = [word_to_id.get(word, unk_id) for word in start_text.split(' ')]
        tokens = np.array([idxs] * batch_size)

    hidden = None
    for _ in range(size):
        softmax, hidden = next(next_proba_gen([tokens[:, -1]], model, hidden_state=hidden))
        softmax = softmax
        if temperature != 1.0:
            softmax = np.float_power(softmax, 1.0/temperature)
            softmax /= softmax.sum(axis=1)

        new_tokens = []
        for sftmx in softmax:
            idx = np.random.choice(list(range(len(sftmx))), p=sftmx)
            new_tokens.append([idx])

        new_tokens = np.array(new_tokens)
        tokens = np.concatenate([tokens, new_tokens], axis=1)

    return [' '.join([id_to_word[idx] for idx in t]) for t in tokens]

class Evaluator:
    def __init__(self):
        pass

    def train_unigram_model(self, token_list, word_to_id):
        vocab_size = len(word_to_id)
        counter = Counter(token_list)
        counts = np.array([counter[i] for i in range(vocab_size)])
        unigram_probs = normalize(counts)
        return unigram_probs

    def check_softmax(self, softmax):
        if (softmax < 0).any():
            print("Some probabilities are <0")

        if (softmax > 1).any():
            print("Some probabilities are >1")

        eps = 1e-3
        prob = softmax.sum(axis=-1)

        dontsum = abs(1.0 - prob) > eps
        if dontsum.any():
            raise Exception("Sum of the probabilities isn't equal to 1. Sum: {}".format(prob[dontsum]))

    def predict_probs(self, model, id_to_word, data, unigram_probs, name, top_k=3, bs=100):
        data = np.array(data)
        tail = len(data)//bs*bs
        # X[i+1,j] is the next word after X[i,j]
        X, X_tail = data[0:tail].reshape(bs, -1).T, data[tail:-1].reshape(1, -1).T 
        assert (np.concatenate([X[:, i] for i in range(X.shape[1])] + [X_tail[:, 0]]) == data[:-1]).all()
        Y, Y_tail = data[1:tail+1].reshape(bs, -1).T, data[tail+1:].reshape(1, -1).T

        preds = []
        desc = 'Generate probability of the next word. Dataset: "{}", batch size: {}'.format(name, bs)
        X_protected_it = ProtectedTokenIterator(X)
        for cur_id, next_id, (softmax, _) in zip(tqdm(X, desc=desc), Y, next_proba_gen(X_protected_it, model)):
            X_protected_it.allow_next = True
            self.append_prediction(preds, cur_id, next_id, softmax, unigram_probs, id_to_word, top_k, bs)

        # convert to triple of 1D-arrays, making next column continuation of the previous one
        r = map(lambda lol: np.array(lol).T.ravel(), zip(*preds))

        bs = 1 # process tail with batch size 1
        preds = []
        desc = 'Generate probability of the next word. Dataset: "{}" (tail), batch size: {}'.format(name, bs)
        X_protected_it = ProtectedTokenIterator(X_tail)
        for cur_id, next_id, (softmax, _) in zip(tqdm(X_tail, desc=desc), Y_tail, next_proba_gen(X_protected_it, model)):
            X_protected_it.allow_next = True
            self.append_prediction(preds, cur_id, next_id, softmax, unigram_probs, id_to_word, top_k, bs)

        rtail = map(lambda lol: np.array(lol).T.ravel(), zip(*preds))
        return chain(zip(*r), zip(*rtail))  # convert to iterable over triples

    def append_prediction(self, preds, cur_id, next_id, softmax, unigram_probs, id_to_word, top_k, bs):
        assert softmax.shape == (bs, len(id_to_word),)
        self.check_softmax(softmax)

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

    def sampling(self, size, 
                 ptb_path=PTB_PATH,
                 start_text=None, 
                 batch_size=20,
                 temperature=1.0, 
                 pretrained_model=None):
        """
            Parameters:
                size (int): - number of generated tokens
                start_text (str): - list of tokens separated by a space
                pretrained_model (str): - path to pretrained model

            Returns:
                generated text (str)
        """

        raw_data = load_dataset(ptb_path)
        train_data, dev_data, test_data, word_to_id, id_to_word = raw_data

        if pretrained_model is None:
            print(datetime_str(), 'Training model ...')
            model = train(train_data, word_to_id, id_to_word)
            print(datetime_str(), 'Training model finished.')
        else:
            model = train(train_data, word_to_id, id_to_word, pretrained_model=pretrained_model)

        unigram_probs = self.train_unigram_model(train_data, word_to_id)

        generated_texts = ancestral_sampling_batch(model, 
                                                   unigram_probs, 
                                                   word_to_id, 
                                                   id_to_word, 
                                                   size=size, 
                                                   batch_size=batch_size,
                                                   start_text=start_text, 
                                                   temperature=temperature)

        print('\n'.join(generated_texts))

    def evaluate(self, ptb_path=PTB_PATH):
        raw_data = load_dataset(ptb_path)
        train_data, dev_data, test_data, word_to_id, id_to_word = raw_data

        print(datetime_str(), 'Training model ...')
        model = train(train_data, word_to_id, id_to_word)
        print(datetime_str(), 'Training model finished.')
        unigram_probs = self.train_unigram_model(train_data, word_to_id)

        allpreds = [['prev', 'pred1', 'pred2', 'pred3', 
                     'true_prob', 'true_rank', 'kl_uniform', 'kl_unigram']]

        print(datetime_str(), 'Testing model ...')
        allpreds.extend(self.predict_probs(model, id_to_word, train_data, unigram_probs, 'train'))
        allpreds.extend(self.predict_probs(model, id_to_word, dev_data, unigram_probs, 'valid'))
        allpreds.extend(self.predict_probs(model, id_to_word, test_data, unigram_probs, 'test'))
        print(datetime_str(), 'Testing model finished.')

        save_preds(allpreds, preds_fname=PREDS_FNAME)

        scores = score_preds(PREDS_FNAME, ptb_path)

        for method_name, method_scores in scores.items():
            for metric, value in method_scores.items():
                print('{} {}: {}'.format(method_name.capitalize(), metric, value))
            print()


if __name__=='__main__':
    fire.Fire(Evaluator)
