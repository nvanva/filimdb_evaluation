from fire import Fire
from time import time
from pathlib import Path

import classifier  # classifier.py should be in the same directory
import score

PREDS_FNAME = Path(__file__).with_name("preds.tsv")

def load_ds(ds_name: str):
    return score.load_dataset_fast(ds_name, parts=score.SCORED_PARTS)


def pretrain(ds_name, module, part2xy, transductive):
    train_ids, train_texts, train_labels = part2xy['train']
    _, train_unlabeled_texts, _ = score.load_dataset_fast(ds_name, parts=('train_unlabeled',))['train_unlabeled']

    if transductive:
        all_texts = list(text for _, text, _ in part2xy.values())
    else:
        all_texts = [train_texts, train_unlabeled_texts]

    total_texts = sum(len(text) for text in all_texts)
#    print('\nPretraining classifier on %d examples from %s; transductive=%s' % (total_texts, pretrain_parts,transductive))
    print('\nPretraining classifier on %d examples' % total_texts)
    st = time()
    params = module.pretrain(all_texts)
    print('Classifier pretrained in %.2fs' % (time() - st))
    return params
   

def train(module, part2xy, pretrain_params):
    train_ids, train_texts, train_labels = part2xy['train']
    print('\nTraining classifier on %d examples from train set ...' % len(train_texts))
    st = time()
    if pretrain_params is not None:  
        params = module.train(train_texts, train_labels, pretrain_params)
    else:
        params = module.train(train_texts, train_labels)
    print('Classifier trained in %.2fs' % (time() - st))
    return params
 

def test(module, part2xy, params):
    allpreds = []
    metrics = {}
    for part, (ids, x, y) in part2xy.items():
        print('\nClassifying %s set with %d examples ...' % (part, len(x)))
        st = time()
        preds = module.classify(x, params)
        print('%s set classified in %.2fs' % (part, time() - st))
        allpreds.extend(zip(ids, preds))

        if y is None:
            print('no labels for %s set' % part)
        else:
            metrics[part] = score.score(preds, y)
    return metrics, allpreds


def save_preds(allpreds):
    score.save_preds(allpreds, preds_fname=PREDS_FNAME)
    print('\nChecking saved predictions ...')
    score.score_preds(preds_fname=PREDS_FNAME, data_dir='FILIMDB')


def main(ds_name='FILIMDB', transductive: bool = False):
    ds = load_ds(ds_name)
    if hasattr(classifier, 'pretrain'):
        pretrain_params = pretrain(ds_name, classifier, ds, transductive)
    else:
        pretrain_params = None

    params = train(classifier, ds, pretrain_params) 
    metrics, preds = test(classifier, ds, params)
    save_preds(preds)


if __name__ == '__main__':
    Fire(main)
