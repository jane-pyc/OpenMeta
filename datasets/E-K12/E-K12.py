import os
import sys
import pickle
import logging
import argparse
import datetime
import numpy as np
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score, accuracy_score,
    roc_auc_score, f1_score, matthews_corrcoef,
    classification_report
)


class LabelLoader:
    def __init__(self, fpath):
        self.records = self._read(fpath)

    def _read(self, fpath):
        store = {}
        with open(fpath) as fp:
            for ln in fp:
                cols = ln.strip().split("\t")
                store[int(cols[0])] = {
                    "tag": cols[1], "desc": cols[2],
                    "grp": cols[3], "aux": cols[4]
                }
        return store

    def grp_of(self, key):
        return self.records.get(key, {}).get("grp", "None")

    def is_paired(self, a, b):
        if a == b:
            return False
        ga, gb = self.grp_of(a), self.grp_of(b)
        return ga != "None" and gb != "None" and ga == gb


class FeatureReader:
    def __init__(self, pkl_path):
        self.pkl_path = pkl_path

    def load(self):
        with open(self.pkl_path, 'rb') as fp:
            data = pickle.load(fp)
        emb_list, id_list = [], []
        for rec in data:
            emb_list.append(rec['embeds'])
            id_list.append(rec['prot_ids'])
        return np.stack(emb_list), np.stack(id_list)


class PairwiseEval:
    def __init__(self, label_loader):
        self.label_loader = label_loader
        self.clf = RandomForestClassifier()

    def make_pairs(self, emb_arr, id_arr):
        xs, ys = [], []
        n_samples = emb_arr.shape[0]
        for idx in range(n_samples):
            seq_emb = emb_arr[idx]
            seq_ids = id_arr[idx]
            for i, ki in enumerate(seq_ids):
                for j, kj in enumerate(seq_ids):
                    if ki <= 0 or kj <= 0:
                        continue
                    if j <= i or j - i >= 2:
                        continue
                    pair_feat = np.concatenate([seq_emb[i], seq_emb[j]])
                    xs.append(pair_feat)
                    ys.append(1 if self.label_loader.is_paired(ki, kj) else 0)
        return np.array(xs), np.array(ys)

    def run_cv(self, xs, ys, log_fn):
        kfold = KFold(n_splits=5, shuffle=True, random_state=16)
        agg_y, agg_p = [], []

        for fid, (tr, te) in enumerate(kfold.split(xs)):
            xtr, xte = xs[tr], xs[te]
            ytr, yte = ys[tr], ys[te]
            self.clf.fit(xtr, ytr)
            prob = self.clf.predict_proba(xte)[:, 1]
            pred = (prob > 0.5).astype(int)
            res = {
                "acc": accuracy_score(yte, pred),
                "f1": f1_score(yte, pred),
                "auc": roc_auc_score(yte, pred),
                "mcc": matthews_corrcoef(yte, pred),
                "ap": average_precision_score(yte, prob)
            }
            print(f"\n[Fold {fid + 1}]")
            for nm, vl in res.items():
                print(f"  {nm}: {vl:.4f}")
                log_fn(f"{nm}: {vl}")
            agg_y.append(yte)
            agg_p.append(prob)

        yfinal = self.clf.predict(xte)
        rpt = classification_report(yte, yfinal, output_dict=True)
        print("\n" + "=" * 50)
        print(classification_report(yte, yfinal))
        final_res = {
            "final_acc": accuracy_score(yte, yfinal),
            "macro_p": rpt['macro avg']['precision'],
            "macro_r": rpt['macro avg']['recall'],
            "macro_f1": rpt['macro avg']['f1-score'],
            "wt_p": rpt['weighted avg']['precision'],
            "wt_r": rpt['weighted avg']['recall'],
            "wt_f1": rpt['weighted avg']['f1-score']
        }
        for nm, vl in final_res.items():
            print(f"{nm}: {vl:.4f}")
            log_fn(f"{nm}: {vl}")
        print("=" * 50)


def init_log(base):
    ts = datetime.datetime.now().strftime("%d-%m-%Y-%H:%M:%S")
    if base is None:
        base = f"./eval_output/-{ts}"
    os.makedirs(base, exist_ok=True)
    logfile = f"{base}/run.log"
    logging.basicConfig(
        format='%(asctime)s - %(message)s', level=logging.INFO,
        handlers=[logging.FileHandler(logfile), logging.StreamHandler()]
    )
    return base, logfile


def main(opts):
    np.random.seed(42)
    base, logfile = init_log(opts.output_dir)
    logging.info(f"Output: {base}")
    logging.info(f"Log: {logfile}")
    logging.info(f"Cmd: {' '.join(sys.argv)}")
    logging.info(f"Feature: {opts.feature_file}")
    logging.info(f"Label: {opts.label_file}")

    labels = LabelLoader(opts.label_file)
    reader = FeatureReader(opts.feature_file)
    evaluator = PairwiseEval(labels)

    embs, ids = reader.load()
    logging.info(f"Loaded {embs.shape[0]} samples, emb_dim={embs.shape[-1]}")

    xs, ys = evaluator.make_pairs(embs, ids)
    logging.info(f"Generated {xs.shape[0]} pairs, feat_dim={xs.shape[-1]}")
    logging.info(f"Positive: {ys.sum()}, Negative: {len(ys) - ys.sum()}")

    evaluator.run_cv(xs, ys, logging.info)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument('-f', '--feature_file', type=str, required=True,
                    help='Path to .pkl feature file')
    ap.add_argument('-l', '--label_file', type=str, required=True,
                    help='Path to .annot label file')
    ap.add_argument('-o', '--output_dir', type=str, default=None,
                    help='Output directory for logs')
    main(ap.parse_args())
