from typing import List, Any
from random import random


def count_labels(labels: List):
    return {
        unique_label: sum(1 for label in labels if label == unique_label)
        for unique_label in set(labels)
    }


def train(
        train_texts: List[str],
        train_labels: List[str],
        pretrain_params: Any = None) -> Any:
    """
    Trains classifier on the given train set represented as parallel lists of texts and corresponding labels.
    :param train_texts: a list of texts (str objects), one str per example
    :param train_labels: a list of labels, one label per example
    :param pretrain_params: parameters that were learned at the pretrain step
    :return: learnt parameters, or any object you like (it will be passed to the classify function)
    """

    # ############################ REPLACE THIS WITH YOUR CODE #############################
    label2cnt = count_labels(train_labels)  # count labels
    print('Labels counts:', label2cnt)
    train_size = sum(label2cnt.values())
    label2prob = {label: cnt / train_size for label, cnt in label2cnt.items()}  # calculate p(label)
    print(label2prob)
    return {'prior': label2prob}  # this dummy classifier learns prior probabilities of labels p(label)
    # ############################ REPLACE THIS WITH YOUR CODE #############################


def pretrain(texts_list: List[List[str]]) -> Any:
    """
    Pretrain classifier on unlabeled texts. If your classifier cannot train on unlabeled data, skip this.
    :param texts_list: a list of list of texts (str objects), one str per example.
        It might be several sets of texts, for example, train and unlabeled sets.
    :return: learnt parameters, or any object you like (it will be passed to the train function)
    """
    # ############################ PUT YOUR CODE HERE #######################################
    return None


def classify(texts: List[str], params: Any) -> List[str]:
    """
    Classify texts given previously learnt parameters.
    :param texts: texts to classify
    :param params: parameters received from train function
    :return: list of labels corresponding to the given list of texts
    """
       
    # ############################ REPLACE THIS WITH YOUR CODE #############################
    def random_label(_label2prob):
        rand = random()  # random value in [0.0, 1.0) from uniform distribution
        for label, prob in _label2prob.items():
            rand -= prob
            if rand <= 0:
                return label

    label2prob = params['prior']
    res = [random_label(label2prob) for _ in texts]  # this dummy classifier returns random labels from p(label)
    print('Predicted labels counts:')
    print(count_labels(res))
    return res
    # ############################ REPLACE THIS WITH YOUR CODE #############################
