import re
from itertools import groupby
from typing import List
from random import randint


def tokenize(sentence: str):
    # ############################ REPLACE THIS WITH YOUR CODE #############################
    return re.findall(r"(\w+)", sentence)
    # ############################ REPLACE THIS WITH YOUR CODE #############################


def cluster_target_word_instances(
    sentences: List[str],
    target_positions: List[str],
    target_word_lemma: str,
    language: str = "ru",
) -> List[str]:
    """
    Contexts clustering according to the meaning of the target_word.
    :param sentences: list of sentences that contain the same target word (target_word_lemma)
    :param target_positions: list of target word positions
        Position example: '110-114'
            Where '110-114' - indexes of the first and last characters of the target word
        Another example: '17-22,86-91' - positions for different target words are separated by a comma
            Where '17-22' - indexes of the first and last characters of the first target word
            And '86-91' - indexes of the first and last characters of the second target word
    :param target_word_lemma: target word lemma, you can use it or not
    :param language: language of sentences. For example 'ru'.
    :return: clustering labels
    """
    # ############################ REPLACE THIS WITH YOUR CODE #############################
    for context, pos in zip(sentences, target_positions):
        # bts-rnc doesn't contain multiple target words in the same sentence
        l, r = (int(p) for p in pos.split(",")[0].split("-"))

        # Example of sentence tokenization
        target_word = context[l:r+1]
        left_tokens, right_tokens = tokenize(context[:l]), tokenize(context[r+1:])
        tokens = left_tokens + [target_word] + right_tokens
        target_idx = len(left_tokens)

    # random clusterizer
    return list(randint(0, 1) for _ in range(len(target_positions)))
    # ############################ REPLACE THIS WITH YOUR CODE #############################


def cluster_sentences(
    target_words: List[str],
    sentences: List[str],
    target_positions: List[str],
    language: str = "ru",
) -> List[str]:
    """
    sentences and target_positions are grouped by target_words
    and then clustered according to the meaning of the target word.
    Obtained labels are combined according
    to the initial positions of the instances.
    :param target_words: list of ambiguous words
    :param sentences: list of sentences
    :param target_positions: list of target word positions
    :param language: language of sentences. You can use a specific model for each language.
    :return: clustering labels
    """

    instances = sorted(
        zip(target_words, range(len(target_positions))), key=lambda it: it[0],
    )

    idx2label = dict()
    for target_word, grouped_instances in groupby(instances, lambda it: it[0]):
        grouped_inst_idxs = [idx for _, idx in grouped_instances]
        grouped_sentences = [sentences[idx] for idx in grouped_inst_idxs]
        grouped_target_positions = [target_positions[idx] for idx in grouped_inst_idxs]

        labels = cluster_target_word_instances(
            grouped_sentences, grouped_target_positions, target_word
        )
        for label, idx in zip(labels, grouped_inst_idxs):
            idx2label[idx] = label

    return [idx2label[idx] for idx in range(len(target_positions))]
