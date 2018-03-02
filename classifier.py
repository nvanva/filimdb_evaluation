def train(texts, labels):
    """
    Trains classifier on the given train set represented as parallel lists of texts and corresponding labels.
    :param texts: a list of texts (str objects), one str per example
    :param labels: a list of labels, one label per example
    :return: learnt parameters, or any object you like (it will be passed to the classify function)
    """

    ############################# REPLACE THIS WITH YOUR CODE #############################
    return labels[0]  # this dummy classifier learns single parameter - the label of the first example
    ############################# REPLACE THIS WITH YOUR CODE #############################
   

def classify(texts, params):
    """
    Classify texts given previously learnt parameters.
    :param texts: texts to classify
    :param params: parameters received from train function
    :return: list of labels corresponding the the given list of texts
    """
       
    ############################# REPLACE THIS WITH YOUR CODE #############################
    return [params for _ in texts]  # this dummy classifier uses the label learnt in train() for all examples
    ############################# REPLACE THIS WITH YOUR CODE #############################
