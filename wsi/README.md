# Word Sense Induction tasks

1. Clone this repository:
```
git clone https://github.com/nvanva/filimdb_evaluation.git
```

2. run init.sh to environment and dataset:
```
cd filimdb_evaluation/wsi/
bash init.sh
```

3. create clusterizer.py and write the following functions:
```
def cluster_target_word_instances(tokens_lists, target_idxs, target_word, language):
    """
    Clusters tokens_lists according to the meaning of the target_word.
    :param tokens_lists: lists of tokens
    :param target_idxs: lists of target word indexes
    :param target_word: target word
    :param language: language of sentences
    :return: clustering labels
    """


def cluster_sentences(target_words, tokens_lists, target_idxs, language):
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
```
4. place clusterizer.py in the filimdb_evaluation/wsi/ folder and run evaluate.py. It will score your clusterizer and create file bts-rnc.csv with predictions.
```
python evaluate.py --dataset="bts-rnc"
```
5. Upload bts-rnc.csv to http://compai-msu.info/.
Register for the appropriate competition, you will receive an e-mail with submission instructions.
6. Upload your classifier following instructions at the appropriate Assignment Submission page.
