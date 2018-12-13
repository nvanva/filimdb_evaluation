import os
import argparse
import numpy as np
from time import time
from score import score_preds, save_preds, load_preds
from lm import softmax_generator

PREDS_FNAME = 'preds.tsv'

def check_softmax(softmax):
    e = 1e-3
    prob = softmax.sum()

    if abs(1.0 - prob) > e:
        print("Sum of the probabilities isn't equal to 1")
        print(prob)
        exit(0)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ptb_path', default='PTB', help='Path to PTB data')
    args = parser.parse_args()

    ptb_path = args.ptb_path

    allpreds = []
    for id_to_word, softmax, cur_word, target_word in softmax_generator(ptb_path):

        # Проверка на batch_size=1, num_steps=1
        assert softmax.shape == (1, 1, len(id_to_word))
        assert cur_word.shape == (1, 1)
        assert target_word.shape == (1, 1)

        # Проверка на суммируемость вероятностей к 1
        check_softmax(softmax[0, 0])

        ind_next_word = np.argmax(softmax, axis=2)[0, 0]
        next_prob = softmax[0, 0, target_word[0, 0]]

        pred_word = id_to_word[ind_next_word]
        allpreds.append([next_prob, pred_word, id_to_word[cur_word[0, 0]]])

    save_preds(allpreds, preds_fname=PREDS_FNAME)

    score_preds(PREDS_FNAME, ptb_path)

if __name__=='__main__':
    main()