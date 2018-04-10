# classifier.py should be in the same directory
from time import time
import sys
from score import load_dataset_fast, score, save_preds, score_preds
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


def main(train_timeout=5 * 60, eval_timeout=5 * 60):
    results = {}
    try:
        import classifier
        importlib.reload(classifier)
    except Exception as e:
        print(e)
        results["exception"] = str(e)
        if sys.modules.get("classifier"):
            del sys.modules['classifier']
        return results

    part2xy = load_dataset_fast('FILIMDB')
    train_ids, train_texts, train_labels = part2xy['train']

    print('\nTraining classifier on %d examples from train set ...' % len(train_texts))
    st = time()

    try:
        with time_limit(train_timeout):
            params = classifier.train(train_texts, train_labels)
    except (TimeoutException, ValueError, Exception) as e:
        del sys.modules['classifier']
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
                preds = classifier.classify(x, params)
        except (TimeoutException, ValueError) as e:
            del sys.modules['classifier']
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
    del sys.modules['classifier']
    return results


if __name__ == '__main__':
    main()
