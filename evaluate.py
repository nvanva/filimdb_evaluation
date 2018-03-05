from time import time
from classifier import train, classify  # classifier.py should be in the same directory
from score import load_dataset_fast, score, save_preds, score_preds

PREDS_FNAME = 'preds.tsv'


def main():
    part2xy = load_dataset_fast('FILIMDB')
    train_ids, train_texts, train_labels = part2xy['train']

    print('\nTraining classifier on %d examples from train set ...' % len(train_texts))
    st = time()
    params = train(train_texts, train_labels)
    print('Classifier trained in %.2fs' % (time()-st))

    allpreds = []
    for part, (ids, x, y) in part2xy.items():
        print('\nClassifying %s set with %d examples ...' % (part, len(x)))
        st = time()
        preds = classify(x, params)
        print('%s set classified in %.2fs' % (part, time() - st))
        allpreds.extend(zip(ids, preds))

        if y is None:
            print('no labels for %s set' % part)
        else:
            score(preds, y)

    save_preds(allpreds, preds_fname=PREDS_FNAME)
    print('\nChecking saved predictions ...')
    score_preds(preds_fname=PREDS_FNAME, data_dir='FILIMDB')

if __name__=='__main__':
    main()
