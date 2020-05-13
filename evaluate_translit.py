from fire import Fire
from time import time
import numpy as np

from translit import train, classify
from score_translit import load_dataset, score, save_preds, score_preds, SCORED_PARTS

PREDS_FNAME = 'preds.tsv'
DATA_DIR = './TRANSLIT'


def main():
    top_k = 1
    part2ixy = load_dataset(DATA_DIR, parts=SCORED_PARTS)
    train_ids, train_strings, train_transliterations = part2ixy['train']
    print('\nTraining classifier on %d examples from train set ...' % len(train_strings))
    st = time()
    params = train(train_strings, train_transliterations)
    print('Classifier trained in %.2fs' % (time() - st))

    allpreds = []
    for part, (ids, x, y) in part2ixy.items():
        print('\nClassifying %s set with %d examples ...' % (part, len(x)))
        st = time()
        preds = classify(x, params)
        print('%s set classified in %.2fs' % (part, time() - st))
        count_of_values = list(map(len, preds))
        assert np.all(np.array(count_of_values) == top_k)
        #score(preds, y)
        allpreds.extend(zip(ids, preds))

    save_preds(allpreds, preds_fname=PREDS_FNAME)
    print('\nChecking saved predictions ...')
    score_preds(preds_path=PREDS_FNAME, data_dir=DATA_DIR)


if __name__ == '__main__':
    Fire(main)
