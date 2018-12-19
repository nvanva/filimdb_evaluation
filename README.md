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

5. Upload preds.tsv to http://compai-msu.info/
6. Upload classifier.py to http://mdl.cs.msu.ru


# lm_evaluation

1. Создайте файл lm.py и напишите генератор softmax_generator(ptb_path). Пример можно посмотреть в уже созданном файле lm.py. Там функция для каждого токена из тестового набора считает случайный softmax для того чтобы предсказать следующее слово, и возвращает словарь id_to_word, softmax, текущее слово, и следующее слово, которое мы должны предсказать по текущему. Для случая языковой модели в этом генераторе нужно сделать следующее:
    + Либо загрузить уже обученную модель, либо обучить ее и сделать следующие шаги.
    + batch_size=1 и num_steps=1.
    + Для ptb.test.txt на каждом таймстепе возвращать:
        + id_to_word - его возвращает load_dataset.
        + softmax(распреденеие по словарю).
        + текущий батч - матрица размером (1, 1), в которой единственный элемент, это слово на входе модели.
        + следующий батч, это так же матрица размером (1, 1).
    + То есть батчи которые возвращает реализованная Вами функция batch_generator.

    ```
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
    ```
2. Поместите файл lm.py в одну директорию с evaluate.py, и запустите evaluate.py. На выходе получится итоговая перплексия, и файл pred.tsv.
    ```
        $ python evaluate.py
        или
        $ python evaluate.py --ptb_path='PTB'
    ```
3. Загрузите preds.tsv на http://compai-msu.info/c/ilimdb_sentiment/description.