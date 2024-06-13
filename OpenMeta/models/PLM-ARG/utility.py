import torch
from math import ceil
import numpy as np
from esm import FastaBatchedDataset, pretrained
from torch.utils.data import DataLoader
from sklearn.preprocessing import MultiLabelBinarizer

AA_list = ('G','A','V','L','I','P','F','Y','W','R'
           'S','T','C','M','N','Q','D','E','K','H')

def AA_replace(seq):
    odd_AAs = set()
    for s in seq:
        if s not in AA_list:
            odd_AAs.add(s)
    for k in odd_AAs:
        seq = seq.replace(k,'X')        
    return seq
   
def extract(fasta_file, alphabet, model,repr_layers=[32], batch_size=500, max_len = 200):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu') 
    dataset = FastaBatchedDataset.from_file(fasta_file)
    seq_num = len(dataset.sequence_labels)
    for i in range(seq_num):
        dataset.sequence_strs[i] = AA_replace(dataset.sequence_strs[i])
    N = ceil(seq_num/batch_size)
    batches = []
    for k in range(N):
        n = int(k*batch_size)
        batches.append(list(range(n, int(min(n+batch_size, seq_num)))))
    if torch.cuda.is_available():
        model = model.cuda()
        print("Transferred model to GPU")
    data_loader = DataLoader(dataset, collate_fn=alphabet.get_batch_converter(),batch_sampler=batches)
    print(f"Read {fasta_file} with {len(dataset)} sequences")    
    repr_layers = [min(i, model.num_layers) for i in repr_layers]        
    result = {layer:torch.empty([0, ]) for layer in repr_layers}
    seq_id = []
    with torch.no_grad():
        for batch_idx, (labels, strs, toks) in enumerate(data_loader):
            print(f"Processing {batch_idx + 1} of {len(batches)} batches ({toks.size(0)} sequences)")
            toks = toks[:, :max_len]
            toks=toks.to(device)
            out = model(toks, repr_layers=repr_layers, return_contacts=False)
            seq_id.extend(labels)
            for layer, t  in out['representations'].items():
                for i, label in enumerate(labels):
                    tmp = t[i, 1 : len(strs[i]) + 1].mean(0).unsqueeze(0)
                    result[layer] =  torch.cat((result[layer].to(device=device), tmp),0)
    result = result[repr_layers[0]].detach().cpu().numpy()
    return seq_id, result

def get_label(seq_id, min_seq =50):
    Label_ID = []
    for ID in seq_id:
        protein_id, src, arg_classes = ID.split("|")
        Label_ID.append(arg_classes.split(";"))
    mlb = MultiLabelBinarizer()
    Label_ID = mlb.fit_transform(Label_ID)
    ARG_Category = mlb.classes_
    if(ARG_Category.shape[0] < 2):
        print("Error: The number of category is less than 2!")
        return
    if(ARG_Category.shape[0] > 2):
        arg_freq = Label_ID.sum(axis = 0)
        rare_id = np.where(arg_freq < min_seq)[0]
        nonarg_id = np.where(ARG_Category=="nonARG")[0]
        multi_drug_id = np.where(ARG_Category=="multi-drug")[0]
        others_id = np.where(ARG_Category == "antibiotic without defined classification")[0]
        others_id = np.append(np.append(rare_id, multi_drug_id), others_id)        
        others_arg = Label_ID[:,others_id].sum(axis=1)
        others_arg = np.where(others_arg>0, 1, others_arg)
        nonarg = Label_ID[:,nonarg_id]
        arg = 1-nonarg
        Label_ID = np.delete(Label_ID, np.append(others_id,nonarg_id), axis=1)
        ARG_Category = np.delete(ARG_Category, np.append(others_id,nonarg_id), axis=0)
        Label_ID = np.insert(Label_ID, Label_ID.shape[1], values=[others_arg],axis = 1)
        Label_ID = np.insert(Label_ID, 0, values=[[arg]],axis = 1)
        ARG_Category = np.insert(ARG_Category, ARG_Category.shape[0], "others",axis=0)
    return Label_ID, ARG_Category


    
    
