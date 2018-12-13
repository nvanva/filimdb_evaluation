import numpy as np
from score import load_dataset

def compute_softmax(x):
    exp = np.exp(np.asarray(x))
    return exp / exp.sum(0)

def softmax_generator(ptb_path):
    """
    input:
        path to the PTB data folder.
    return:
        :id_to_word - dict: word index -> word, size -> vocabulary size
        :softmax - shape=(1,1,vocab_size)
        :language model input for the current time step
        :language model output for the current time step
    """

    raw_data = load_dataset(ptb_path)
    _, _, test_data, word_to_id, id_to_word = raw_data

    vocab_size = len(word_to_id)

    for cur_word, next_word in zip(test_data[:-1], test_data[1:]):
        probs = np.random.rand(vocab_size)

        softmax = compute_softmax(probs).reshape([1, 1, vocab_size])

        yield id_to_word, softmax, np.asarray([[cur_word]]), np.asarray([[next_word]])