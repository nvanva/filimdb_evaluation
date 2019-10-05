from time import time
from classifier import train, classify  # classifier.py should be in the same directory
from score import load_dataset_fast, score, save_preds, score_preds, SCORED_PARTS

PREDS_FNAME = 'preds.tsv'


def main():
    try:
        from classifier import pretrain
    except ImportError:
        part2xy = load_dataset_fast('FILIMDB', parts=SCORED_PARTS)
        train_ids, train_texts, train_labels = part2xy['train']
        print('\nTraining classifier on %d examples from train set ...' % len(train_texts))
        st = time()
        params = train(train_texts, train_labels)
        print('Classifier trained in %.2fs' % (time() - st))
    else:
        part2xy = load_dataset_fast('FILIMDB', parts=SCORED_PARTS+('train_unlabeled',))
        train_ids, train_texts, train_labels = part2xy['train']
        _, train_unlabeled_texts, _ = part2xy['train_unlabeled']
        all_texts = train_texts + train_unlabeled_texts 
        
        print('\nPretraining classifier on %d examples' % len(all_texts))
        st = time()
        params = pretrain(all_texts)
        print('Classifier pretrained in %.2fs' % (time() - st))
        print('\nTraining classifier on %d examples from train set ...' % len(train_texts))
        st = time()
        params = train(train_texts, train_labels, params)
        print('Classifier trained in %.2fs' % (time() - st))
        del part2xy["train_unlabeled"]

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


if __name__ == '__main__':
    main()
