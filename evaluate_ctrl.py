# classifier.py should be in the same directory
from time import time
import sys
from score import load_dataset_fast, score, save_preds, score_preds, SCORED_PARTS
import signal
from contextlib import contextmanager
import importlib
import traceback
from fire import Fire

class TimeoutException(Exception): pass

@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Time limit exceeded")

    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)

def handle_exc(e, module, results):
    print('Exception caught:', e)
    results["exception"] = str(e)
    results["exception_full"] = traceback.format_exc()
    if module is not None:
        if sys.modules.get(module.__name__):
            del sys.modules[module.__name__]
    return results 


def main(package, file_name, train_timeout=60 * 30, eval_timeout=60 * 30, transductive=False):
    results = {}
    try:
        module = importlib.import_module(f".{file_name}", package)
        importlib.reload(module)
    except BaseException as e:
       return handle_exc(e, module if 'module' in locals() else None, results)

    if hasattr(module, 'pretrain'):
        part2xy = load_dataset_fast('FILIMDB_hidden', SCORED_PARTS+('train_unlabeled',))
        pretrain_parts = part2xy.keys() if transductive else {'train', 'train_unlabeled'}
        pretrain_texts = [part2xy[part][1] for part in pretrain_parts]
        total_texts = sum(len(text) for text in pretrain_texts)
        print('\nPretraining classifier on %d examples from %s; transductive=%s' % (total_texts, pretrain_parts,transductive))
        st = time()
        try:
            with time_limit(train_timeout):
                pretrain_params = module.pretrain(pretrain_texts)
        except BaseException as e:
            if isinstance(e, TimeoutException):
                results["pretrain_time"] = train_timeout
                print("Timeout on pretraining!" % part)
            return handle_exc(e, module, results)

        pretrain_time = time() - st
        results["pretrain_time"] = pretrain_time
        print('Classifier pretrained in %.2fs' % pretrain_time)
    else:
        part2xy = load_dataset_fast('FILIMDB_hidden', SCORED_PARTS)
        pretrain_params = None

    train_ids, train_texts, train_labels = part2xy['train']
    print('\nTraining classifier on %d examples from train set ...' % len(train_texts))
    st = time()
    try:
        with time_limit(train_timeout):
            params = module.train(train_texts, train_labels, pretrain_params=pretrain_params)
    except BaseException as e:
         if isinstance(e, TimeoutException):
            results["train_time"] = train_timeout
            print("Timeout on training!" % part)
         return handle_exc(e, module, results)

    train_time = time() - st
    results["train_time"] = train_time

    print('Classifier trained in %.2fs' % train_time)

    allpreds = []
    for part, (ids, x, y) in part2xy.items():
        print('\nClassifying %s set with %d examples ...' % (part, len(x)))
        st = time()
        try:
            with time_limit(eval_timeout):
                preds = module.classify(x, params)
        except BaseException as e:
            if isinstance(e, TimeoutException):
                print("Timeout on evaluating %s set!" % part)
                results["eval_on_%s_set_time" % part] = eval_timeout
            return handle_exc(e, module, results)

        eval_time = time() - st
        results["eval_on_%s_set_time" % part] = eval_time
        print('%s set classified in %.2fs' % (part, eval_time))
        allpreds.extend(zip(ids, preds))

        if y is None:
            print('no labels for %s set' % part)
        else:
            acc = score(preds, y)
            results["eval_on_%s_set_acc" % part] = acc
    del sys.modules[module.__name__]
    return results


if __name__ == '__main__':
    Fire(main)
