import os
import argparse
import subprocess
import numpy as np
import pickle as pk
from sklearn.decomposition import PCA

parser = argparse.ArgumentParser()
parser.add_argument('--emb_pkl', type=str, required=True, help='Input embedding pkl file')
parser.add_argument('--meta_tsv', type=str, required=True, help='Input metadata tsv file')
parser.add_argument('--out_dir', type=str, required=True, help='Output directory')
parser.add_argument('--max_seq_len', type=int, default=30, help='Max sequence length')
parser.add_argument('--emb_dim', type=int, default=1281, help='Embedding dimension')
parser.add_argument('--label_dim', type=int, default=100, help='Label dimension')
parser.add_argument('--pca_components', type=int, default=99, help='PCA components')
args = parser.parse_args()

subprocess.call(f"mkdir -p {args.out_dir}", shell=True)

with open(args.emb_pkl, 'rb') as f:
    esm_embs = pk.load(f)

embs = []
all_prot_ids = []
for key, val in esm_embs:
    all_prot_ids.append(key.split(" ")[0])
    embs.append(val)
embs = np.array(embs, dtype=np.float16)

root_dir = os.path.dirname(args.emb_pkl)
split_file_name = os.path.basename(args.emb_pkl).split('.esm')[0]

embs_mean = np.mean(embs, 0)
embs_std = np.std(embs, 0)
normalized_embs = (embs - embs_mean) / embs_std

pca_model = PCA(n_components=args.pca_components, whiten=True)
all_labels = pca_model.fit_transform(normalized_embs)

batch = []
prot_to_id = {}
global_idx = 1

with open(args.meta_tsv, 'r') as f:
    for line in f:
        rec = {}
        embeds = np.zeros((args.max_seq_len, args.emb_dim), dtype=np.float16)
        label_embeds = np.zeros((args.max_seq_len, args.label_dim), dtype=np.float16)
        prot_ids = np.zeros(args.max_seq_len, dtype=int)
        attention_mask = np.zeros(args.max_seq_len, dtype=int)

        elems = line.strip().split(",")
        prots_in_contig = elems[1].split(";")

        for ind, prot_id in enumerate(prots_in_contig):
            pid = prot_id[:]
            prot_index = all_prot_ids.index(pid)
            emb = normalized_embs[prot_index]
            label = all_labels[prot_index]

            emb_o = np.append(emb, 0.5)
            label_o = np.append(label, 0.5)

            embeds[ind] = emb_o
            label_embeds[ind] = label_o
            prot_to_id[global_idx] = pid
            prot_ids[ind] = global_idx
            global_idx += 1
            attention_mask[ind] = 1

        rec['prot_ids'] = prot_ids
        rec['embeds'] = embeds
        rec['label_embeds'] = label_embeds
        rec['attention_mask'] = attention_mask
        batch.append(rec)

batch = np.array(batch)

with open(os.path.join(args.out_dir, "train.pkl"), "wb") as f:
    pk.dump(batch, f)

with open(os.path.join(args.out_dir, "prot_index_dict.pkl"), "wb") as f:
    pk.dump(prot_to_id, f)

print(f"Saved to {args.out_dir}, total samples: {len(batch)}, total proteins: {global_idx - 1}")
