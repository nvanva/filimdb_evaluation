# imdb_evaluation

1. Clone this repository:
```
git clone https://github.com/nvanva/filimdb_evaluation.git
```

2. run init.sh to prepare dataset:
```
./init.sh
```

3. create classifier.py and write the following functions:
```
def train(texts, labels):
    """
    Trains classifier on the given train set represented as parallel lists of texts and corresponding labels.
    :param texts: a list of texts (str objects), one str per example
    :param labels: a list of labels, one label per example
    :return: learnt parameters, or any object you like (it will be passed to the classify function) 
    """

def classify(texts, params):
    """
    Classify texts given previously learnt parameters.
    :param texts: texts to classify
    :param params: parameters received from train function
    :return: list of labels corresponding the the given list of texts
    """
```
4. place classifier.py in the same folder as evaluate.py and run evaluate.py. It will score your classifier and create file preds.tsv with predictions.
```
python evaluate.py
```

5. Upload preds.tsv to http://compai-msu.info/c/ilimdb_sentiment/description.

6. Upload classifier.py to http://mdl.cs.msu.ru, Assignment 1 Submission.
