# usage: python batch_data.py <output.esm.embs.pkl> <contig_to_prots.tsv> <output_dir> 
import os
GPU_NUMBER = [0,1]
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(s) for s in GPU_NUMBER])
import sys
import numpy as np
import pickle as pk
import subprocess

emb_f = sys.argv[1]
contig_to_prots_f = sys.argv[2]
output_dir = sys.argv[3]
subprocess.call(f"mkdir -p {output_dir}", shell=True)

contigs_to_prots_file = open(contig_to_prots_f, "r")
emb_file = open(emb_f, 'rb')
esm_embs = pk.load(emb_file)
embs = []
all_prot_ids = []
for key,val in esm_embs:
    all_prot_ids.append(key.split(" ")[0])
    embs.append(val)
emb_file.close()
embs = np.array(embs, dtype = np.float16)

root_dir = os.path.dirname(emb_f)
split_file_name = os.path.basename(emb_f).split('.esm')[0]

norm_factors_f = os.path.join(root_dir,split_file_name+"_norm.pkl")
pca_pkl_f = os.path.join(root_dir,split_file_name+"_pca.pkl")
normlized_embs_f = os.path.join(root_dir,split_file_name+"_normlized.pkl")

embs_mean = np.mean(embs,0)
embs_std = np.std(embs,0)
embs_dict = {'mean': embs_mean, 'std': embs_std}
normalized_embs = (embs-embs_mean) / embs_std

from sklearn.decomposition import PCA
PCA_LABEL = PCA(n_components=99,whiten=True)
all_labels = PCA_LABEL.fit_transform(normalized_embs)
MAX_SEQ_LENGTH = 30 
EMB_DIM = 1281
LABEL_DIM = 100 
counter = 0
batch = []
index = 0
prot_ids = []
prot_to_id = {}
i = 1
count = 0
for line in contigs_to_prots_file:
    b={}
    embeds = np.zeros((MAX_SEQ_LENGTH, EMB_DIM), dtype=np.float16)
    label_embeds=np.zeros((MAX_SEQ_LENGTH, LABEL_DIM), dtype=np.float16)
    prot_ids =  np.zeros(MAX_SEQ_LENGTH, dtype =int)
    attention_mask = np.zeros(MAX_SEQ_LENGTH, dtype =int)
    elems = line.strip().split(",")
    prots_in_contig = elems[1].split(";")
    for ind, prot_id in enumerate(prots_in_contig):
        ori = '+'
        pid = prot_id[:]
        # pdb.set_trace()
        prot_index = all_prot_ids.index(pid)
        emb = normalized_embs[prot_index]
        label = all_labels[prot_index]
        if ori == "+":
            emb_o = np.append(emb,0.5)
            label_o = np.append(label,0.5)
        else:
            emb_o = np.append(emb,-0.5)
            label_o = np.append(label,-0.5)
        embeds[ind] = emb_o 
        label_embeds[ind] = label_o 
        prot_to_id[i] = pid
        
        prot_ids[ind] = i
        i+=1
        attention_mask[ind] = 1 
        count +=1
    b['prot_ids'] = prot_ids
    b['embeds'] = embeds
    b['label_embeds'] = label_embeds
    b['attention_mask'] = attention_mask
    batch.append(b)
batch = np.array(batch)
f = open(output_dir+f"/train.pkl", "wb")
pk.dump(batch,f)
f.close()

f = open(output_dir+f"/prot_index_dict.pkl", "wb")
pk.dump(prot_to_id, f)
f.close()