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

    eps = 1e-5
    prob = softmax.sum()

    if abs(1.0 - prob) > eps:
        raise Exception("Sum of the probabilities isn't equal to 1. Sum: {}".format(prob))

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

def predict_probs(model, id_to_word, data, name, top_k=3):
    test_token_gen = ProtectedTokenIterator(data)

    preds = []
    desc = 'Generate probability of the next word. Dataset: "{}"'.format(name)
    for prev_id, next_id, softmax in zip(tqdm(data[:-1], desc=desc), data[1:], next_proba_gen(test_token_gen, model)):
        test_token_gen.allow_next = True

        # Check batch_size==1 and num_steps==1
        assert softmax.shape == (len(id_to_word),)

        check_softmax(softmax)

        true_prob = softmax[next_id]
        true_rank = len(np.where(softmax > true_prob)[0])
        
        top_k_idxs = np.argpartition(-softmax, top_k)[: top_k]
        sorted_top_k_idxs = top_k_idxs[np.argsort(-softmax[top_k_idxs])]
        top_k_words = [id_to_word[idx] for idx in sorted_top_k_idxs]

        preds.append([id_to_word[prev_id]] + top_k_words + [true_prob, true_rank])

    return preds

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ptb_path', default='PTB', help='Path to PTB data')
    args = parser.parse_args()

    raw_data = load_dataset(args.ptb_path)
    train_data, dev_data, test_data, word_to_id, id_to_word = raw_data

    model = train(train_data, word_to_id, id_to_word)

    allpreds = [['prev', 'pred1', 'pred2', 'pred3', 'true_prob', 'true_rank']]
    allpreds.extend(predict_probs(model, id_to_word, train_data, 'train'))
    allpreds.extend(predict_probs(model, id_to_word, dev_data, 'valid'))
    allpreds.extend(predict_probs(model, id_to_word, test_data, 'test'))

    save_preds(allpreds, preds_fname=PREDS_FNAME)

    scores = score_preds(PREDS_FNAME, args.ptb_path)

    for method_name, method_scores in scores.items():
        for metric, value in method_scores.items():
            print('{} {}: {}'.format(method_name.capitalize(), metric, value))
        print()

    # comments = []
    # for score in ['perplexity', 'hit@10', 'avg_rank']:
    #     for part in ['train', 'valid']:
    #         comments.append('{}_{}: {:.3f}'.format(part, score, scores[part][score]))
    # comment = ', '.join(comments)
    # print(comment)

if __name__=='__main__':
    main()