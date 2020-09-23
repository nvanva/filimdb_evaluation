import sys
import logging
from fire import Fire
from pathlib import Path

FILIMDB_PATH = Path(__file__).resolve().parent.parent
sys.path.append(str(FILIMDB_PATH))

from wsi.clusterizer import cluster_sentences
from wsi.score import score_preds, load_dataset, save_preds, score_loaded


def evaluate(dataset: str = "bts-rnc"):
    logging.basicConfig(
        format="%(asctime)-16s %(message)s",
        level=logging.INFO,
        datefmt="%d/%m-%H:%M:%S",
    )

    logging.info(f"Loading '{dataset}' dataset")
    part2data, language = load_dataset(dataset)
    logging.info(f"'{language}' dataset '{dataset}' loaded. It consists of "
                 f"{len(part2data)} parts: {list(part2data.keys())}")
    idx2label = dict()

    num_instances = 0
    for part, data in part2data.items():
        context_idxs, target_words, sentences, target_positions = data
        num_instances += len(target_words)
        logging.info(f"Clustering {len(target_words)} instances of '{part}' part")
        labels = cluster_sentences(
            target_words, sentences, target_positions, language
        )
        logging.info(f"Part '{part}' clustered")
        assert len(labels) == len(context_idxs), \
            f"You should provide a label for each instance: " \
            f"provided number of labels - {len(labels)}, " \
            f"number of instances - {len(context_idxs)}"
        for idx, label in zip(context_idxs, labels):
            idx2label[idx] = label

    assert num_instances == len(idx2label)

    logging.info(f"Scoring train predictions")
    scores = score_loaded(dataset, idx2label, parts=("train",))
    logging.info(f"Scoring done. Scores: {scores}")

    preds_fname = Path(__file__).resolve().with_name(f"{dataset}-test").with_suffix(".tsv.tar.gz")
    save_preds(idx2label, dataset, preds_fname=preds_fname, parts=("test",))
    logging.info(f"Test predictions saved to '{preds_fname}'")


if __name__ == "__main__":
    Fire(evaluate)
