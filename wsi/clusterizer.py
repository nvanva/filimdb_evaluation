from itertools import groupby
from typing import List
from random import randint


def cluster_target_word_instances(
    tokens_lists: List[List[str]],
    target_idxs: List[int],
    target_word: str,
    language: str = "ru",
) -> List[str]:
    """
    Clusters tokens_lists according to the meaning of the target_word.
    :param tokens_lists: lists of tokens
    :param target_idxs: lists of target word indexes
    :param target_word: target word
    :param language: language of sentences
    :return: clustering labels
    """
    # ############################ REPLACE THIS WITH YOUR CODE #############################
    # random clusterizer
    return list(randint(0, 1) for _ in range(len(target_idxs)))
    # ############################ REPLACE THIS WITH YOUR CODE #############################


def cluster_sentences(
    target_words: List[int],
    tokens_lists: List[List[str]],
    target_idxs: List[int],
    language: str = "ru",
) -> List[str]:
    """
    tokens_lists and target_idxs are grouped by target_words
    and then clustered according to the meaning of the target word.
    Obtained labels are combined according
    to the initial positions of the instances.
    :param target_words: list of ambiguous words
    :param tokens_lists: list of sentences that are represented as lists of tokens
    :param target_idxs: list of target word indexes
    :param language: language of sentences. You can use a specific model for each language.
    :return: clustering labels
    """

    instances = sorted(
        zip(target_words, range(len(target_idxs))), key=lambda it: it[0],
    )

    idx2label = dict()
    for target_word, grouped_instances in groupby(instances, lambda it: it[0]):
        grouped_inst_idxs = [idx for _, idx in grouped_instances]
        grouped_tokens_lists = [tokens_lists[idx] for idx in grouped_inst_idxs]
        grouped_target_idxs = [target_idxs[idx] for idx in grouped_inst_idxs]

        labels = cluster_target_word_instances(
            grouped_tokens_lists, grouped_target_idxs, target_word
        )
        for label, idx in zip(labels, grouped_inst_idxs):
            idx2label[idx] = label

    return [idx2label[idx] for idx in range(len(target_idxs))]
