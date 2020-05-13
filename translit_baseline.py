from typing import List, Any
from random import random
import collections as col

def train(
        train_source_strings: List[str],
        train_target_strings: List[str]) -> Any:
    """
    Trains transliretation model on the given train set represented as
    parallel list of input strings and their transliteration via labels.
    :param train_source_strings: a list of strings, one str per example
    :param train_target_strings: a list of strings, one str per example
    :return: learnt parameters, or any object you like (it will be passed to the classify function)
    """

    # ############################ REPLACE THIS WITH YOUR CODE #############################
    ngram_lvl = 3
    def obtain_train_dicts(train_source_strings, train_target_strings,
                            ngram_lvl):
        ngrams_dict = col.defaultdict(lambda: col.defaultdict(int))
        for src_str,dst_str in zip(train_source_strings,
                                        train_target_strings):
            try:
                src_ngrams = [src_str[i:i+ngram_lvl] for i in
                                range(len(src_str)-ngram_lvl+1)]
                dst_ngrams = [dst_str[i:i+ngram_lvl] for i in
                                range(len(dst_str)-ngram_lvl+1)]
            except TypeError as e:
                print(src_ngrams, dst_ngrams)
                print(e)
                raise StopIteration
            for src_ngram in src_ngrams:
                for dst_ngram in dst_ngrams:
                    ngrams_dict[src_ngram][dst_ngram] += 1
        return ngrams_dict
        
    ngrams_dict = col.defaultdict(lambda: col.defaultdict(int))
    for nl in range(1, ngram_lvl+1):
        ngrams_dict.update(
            obtain_train_dicts(train_source_strings,
                            train_target_strings, nl))
    return ngrams_dict 
    # ############################ REPLACE THIS WITH YOUR CODE #############################


def classify(strings: List[str], params: Any) -> List[str]:
    """
    Classify strings given previously learnt parameters.
    :param strings: strings to classify
    :param params: parameters received from train function
    :return: list of lists of predicted transliterated strings
      (for each source string -> [top_1 prediction, .., top_k prediction]
        if it is possible to generate more than one, otherwise
        -> [prediction])
        corresponding to the given list of strings
    """
       
    # ############################ REPLACE THIS WITH YOUR CODE #############################
    def predict_one_sample(sample, train_dict, ngram_lvl=1):
        ngrams = [sample[i:i+ngram_lvl] for i in
 range(0,(len(sample) // ngram_lvl * ngram_lvl)-ngram_lvl+1, ngram_lvl)] +\
                 ([] if len(sample) % ngram_lvl == 0 else
                    [sample[-(len(sample) % ngram_lvl):]])
        prediction = ''
        for ngram in ngrams:
            ngram_dict = train_dict[ngram]
            if len(ngram_dict.keys()) == 0:
                prediction += '?'*len(ngram)
            else:
                prediction += max(ngram_dict, key=lambda k: ngram_dict[k])
        return prediction 
    
    ngram_lvl = 3
    predictions = []
    ngrams_dict = params
    for string in strings:
        top_1_pred = predict_one_sample(string, ngrams_dict,
                                                ngram_lvl)
        predictions.append([top_1_pred])
    return predictions
    # ############################ REPLACE THIS WITH YOUR CODE #############################
