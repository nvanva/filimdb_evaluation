from evaluate import load_dataset_fast, score, load_preds


def main():
    preds_dir = '.'

    part2xy = load_dataset_fast('FILIMDB_hidden', parts=['train','dev','test'])
    for part, (true_ids, _, true_y) in part2xy.items():
        pred_ids, pred_y = load_preds(part, dir=preds_dir)

        pred_dict = {i: y for i, y in zip(pred_ids, pred_y)}
        pred_y = [pred_dict[i] for i in true_ids]
        score(pred_y, true_y)


if __name__=='__main__':
    main()