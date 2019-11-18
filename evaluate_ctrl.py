# classifier.py should be in the same directory
from time import time
import sys
from score import load_dataset_fast, score, save_preds, score_preds, SCORED_PARTS
import signal
from contextlib import contextmanager
import importlib


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


def main(package, file_name, train_timeout=60 * 30, eval_timeout=60 * 30):
    results = {}
    try:
        module = importlib.import_module(f".{file_name}", package)
        importlib.reload(module)
    except Exception as e:
        print(e)
        results["exception"] = str(e)
        if sys.modules.get("classifier"):
            del sys.modules['classifier']
        return results

    part2xy = load_dataset_fast('FILIMDB_hidden', SCORED_PARTS)
    train_ids, train_texts, train_labels = part2xy['train']

    print('\nTraining classifier on %d examples from train set ...' % len(train_texts))
    st = time()

    try:
        with time_limit(train_timeout):
            params = module.train(train_texts, train_labels)
    except (TimeoutException, ValueError, Exception) as e:
        del sys.modules[module.__name__]
        print(e)
        if isinstance(e, TimeoutException):
            results["train_time"] = train_timeout
        results["exception"] = str(e)
        return results

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
        except (TimeoutException, ValueError) as e:
            del sys.modules[module.__name__]
            if isinstance(e, TimeoutException):
                print("Timeout on evaluating %s set!" % part)
                results["eval_on_%s_set_time" % part] = eval_timeout
            else:
                print(e)
            results["exception"] = str(e)
            return results

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
    main()
