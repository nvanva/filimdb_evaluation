# Sentiment classification tasks

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
def pretrain(texts):
   """
   Pretrain classifier on unlabeled texts. If your classifier cannot train on unlabeled data, skip this.
   :param texts: a list of texts (str objects), one str per example
   :return: learnt parameters, or any object you like (it will be passed to the train function)
   """
   
def train(texts, labels, pretrain_params=None):
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
5. if you need to pretrain your model on all sets of texts (train, test, dev, unlabeled, dev-b, test-b), use --transductive command-line argument:
```
python evaluate.py --transductive
```
6. Upload preds.tsv to http://compai-msu.info/.
Register for the appropriate competition, you will receive an e-mail with submission instructions.
7. Upload your classifier following instructions at the appropriate Assignment Submission page.


# Language modeling tasks

1. Clone this repository:
```
git clone https://github.com/nvanva/filimdb_evaluation.git
```

2. run init.sh to prepare dataset:
```
./init.sh
```

3. Edit lm.py and write the following functions:

4. Run evaluate_lm.py
    ```
    python evaluate_lm.py evaluate --ptb-path='PTB'
    ```
5. Sampling from lm
    ```
    python evaluate_lm.py sampling --size=20 --start-text='the meaning of life is'
    ```
6. Load preds.tsv to ??? (coming soon).
7. Load lm.py to  http://mdl.cs.msu.ru
