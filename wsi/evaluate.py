import logging
from fire import Fire
from pathlib import Path

from wsi.dataset import load_dataset, save_preds
from wsi.clusterizer import cluster_sentences
from wsi.score import score_preds


def evaluate(dataset: str = "bts-rnc"):
    logging.basicConfig(
        format="%(asctime)-12s %(message)s",
        level=logging.INFO,
        datefmt="%d:%m:%Y %H:%M:%S",
    )

    logging.info(f"Loading '{dataset}' dataset")
    part2data, language = load_dataset(dataset)
    logging.info(f"'{language}' dataset '{dataset}' loaded. It consists of "
                 f"{len(part2data)} parts: {list(part2data.keys())}")
    idx2label = dict()

    num_instances = 0
    for part, data in part2data.items():
        context_idxs, target_words, tokens_lists, target_idxs = data
        num_instances += len(target_words)
        logging.info(f"Clustering {len(target_words)} instances of '{part}' part")
        labels = cluster_sentences(
            target_words, tokens_lists, target_idxs, language
        )
        logging.info(f"Part '{part}' clustered")
        assert len(labels) == len(context_idxs), \
            f"You should provide a label for each instance: " \
            f"provided number of labels - {len(labels)}, " \
            f"number of instances - {len(context_idxs)}"
        for idx, label in zip(context_idxs, labels):
            idx2label[idx] = label

    assert num_instances == len(idx2label)

    preds_fname = Path(__file__).resolve().with_name(dataset).with_suffix(".csv")
    save_preds(idx2label, dataset, preds_fname=preds_fname)
    logging.info(f"Predictions saved to '{preds_fname}'")
    logging.info(f"Scoring '{preds_fname}' predictions")
    scores = score_preds(dataset, preds_fname=preds_fname)
    logging.info(f"Scoring done. Scores: {scores}")


if __name__ == "__main__":
    Fire(evaluate)
