# classifier.py should be in the same directory
from time import time
import sys
from score import load_dataset_fast, score, save_preds, score_preds, SCORED_PARTS
import signal
from contextlib import contextmanager
import importlib
import traceback
from fire import Fire
import evaluate

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


def main(package, file_name, ds_name='FILIMDB_hidden', transductive=False,  train_timeout=60 * 30, test_timeout=60 * 30):
    results = {}
    try:
        module = importlib.import_module(f".{file_name}", package)
        importlib.reload(module)
    except BaseException as e:
       return handle_exc(e, module if 'module' in locals() else None, results)

    ds = evaluate.load_ds(ds_name)
    if hasattr(module, 'pretrain'):
        st = time()
        try:
            with time_limit(train_timeout):
                pretrain_params = evaluate.pretrain(ds_name, module, ds, transductive)
        except BaseException as e:
            if isinstance(e, TimeoutException):
                results["pretrain_time"] = train_timeout
                print("Timeout on pretraining!")
            return handle_exc(e, module, results)

        pretrain_time = time() - st
        results["pretrain_time"] = pretrain_time
        print('Classifier pretrained in %.2fs' % pretrain_time)
    else:
        results["pretrain_time"] = -1
        pretrain_params = None

    st = time()
    try:
        with time_limit(train_timeout):
            params = evaluate.train(module, ds, pretrain_params)
    except BaseException as e:
         if isinstance(e, TimeoutException):
            results["train_time"] = train_timeout
            print("Timeout on training!")
         return handle_exc(e, module, results)

    train_time = time() - st
    results["train_time"] = train_time


    st = time()
    try:
        with time_limit(test_timeout):
            metrics, preds = evaluate.test(module, ds,  params)
    except BaseException as e:
        if isinstance(e, TimeoutException):
            print("Timeout on testing!")
            results["test_time"] = test_timeout
        return handle_exc(e, module, results)

    results["test_time"] = time() - st
    for part, acc in metrics.items():
        results["%s_acc" % part] = acc
    del sys.modules[module.__name__]
    return results


if __name__ == '__main__':
    Fire(main)
