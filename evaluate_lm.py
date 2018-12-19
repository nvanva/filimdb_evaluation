import argparse
import numpy as np
from tqdm import tqdm

from lm import train, next_proba_gen
from score_lm import load_dataset, save_preds, score_preds

PREDS_FNAME = 'preds.tsv'

def check_softmax(softmax):
    if (softmax < 0).any():
        print("Some probabilities are <0")

    if (softmax > 1).any():
        print("Some probabilities are >1")

    e = 1e-7
    prob = softmax.sum()

    if abs(1.0 - prob) > e:
        print("Sum of the probabilities isn't equal to 1")
        print(prob)
        exit(0)


class ProtectedTokenIterator(object):
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ptb_path', default='simple-examples/data', help='Path to PTB data')
    args = parser.parse_args()

    ptb_path = args.ptb_path
    raw_data = load_dataset(ptb_path)
    train_data, dev_data, test_data, word_to_id, id_to_word = raw_data

    model = train(train_data, word_to_id, id_to_word)

    test_token_gen = ProtectedTokenIterator(test_data)

    allpreds = []
    for cur_id, next_id, softmax in zip(tqdm(test_data[:-1]), test_data[1:], next_proba_gen(test_token_gen, model)):
        test_token_gen.allow_next = True

        # Проверка на batch_size=1, num_steps=1
        assert softmax.shape == (len(id_to_word),)

        # Проверка на суммируемость вероятностей к 1
        check_softmax(softmax)

        next_prob = softmax[next_id]
        pred_word = id_to_word[np.argmax(softmax)]
        allpreds.append([next_prob, pred_word, id_to_word[cur_id]])

    save_preds(allpreds, preds_fname=PREDS_FNAME)

    score_preds(PREDS_FNAME, ptb_path)

if __name__=='__main__':
    main()