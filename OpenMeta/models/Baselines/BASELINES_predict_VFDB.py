import os
GPU_NUMBER = [0,1,2,3,4,5,6,7]
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(s) for s in GPU_NUMBER])
import warnings
warnings.filterwarnings('ignore')
from time import *
from Bio import SeqIO
import subprocess
import pandas as pd
import torch
from sklearn.metrics import accuracy_score
from BASELINES_config_VFDB import Config
from BASELINES_network import BASELINES
from Utils import cal
from Utils.combine_fna import combine
import numpy as np
import pickle as pk
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd

def seq_meta_data_process():
    new_data = pd.DataFrame(columns=["contigs", "MGYP_proteins"])
    fasta_path = config.combined_fasta_path
    new_meta_file_path = config.new_meta_file_path
    output_dir = os.path.dirname(new_meta_file_path)
    subprocess.call(f"mkdir -p {output_dir}", shell=True)
    with open(fasta_path,'r') as fa:
        MGYPs = []
        count = 0
        for record in SeqIO.parse(fa,'fasta'):        
            MGYPs.append(record.id)        
            if len(MGYPs) == 30:
                contigs = 'contig'+str(count+1)
                MGYP_names = ";".join(MGYPs)
                new_data.loc[count] = [contigs, MGYP_names]
                count += 1
                MGYPs = []      
        if 15 < len(MGYPs) < 30 :
            contigs = 'contig'+str(count+1)
            MGYP_names = ";".join(MGYPs)
            new_data.loc[count] = [contigs, MGYP_names]
            count += 1
            MGYPs = []
    new_meta_file_dir = new_meta_file_path.split('/')
    new_data.to_csv(new_meta_file_path, header=None, index=False)

def batch_data():
    emb_f = config.freqs_file
    contig_to_prots_f = config.new_meta_file_path
    batch_data_path = config.batch_data_path
    batch_data_prot_index_dict_path = config.batch_data_prot_index_dict_path

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

    embs_mean = np.mean(embs,0)
    embs_std = np.std(embs,0)
    normalized_embs = (embs-embs_mean) / embs_std

    MAX_SEQ_LENGTH = 30 
    EMB_DIM = embs.shape[1]
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
        prot_ids =  np.zeros(MAX_SEQ_LENGTH, dtype =int)
        elems = line.strip().split(",")
        prots_in_contig = elems[1].split(";")
        for ind, prot_id in enumerate(prots_in_contig):            
            pid = prot_id[:]
            prot_index = all_prot_ids.index(pid)
            emb = normalized_embs[prot_index]
            embeds[ind] = emb 
            prot_to_id[i] = pid            
            prot_ids[ind] = i
            i+=1
            count +=1
        b['prot_ids'] = prot_ids
        b['embeds'] = embeds
        batch.append(b)
    batch = np.array(batch)
    f = open(batch_data_path, "wb")
    pk.dump(batch,f)
    f.close()
    f = open(batch_data_prot_index_dict_path, "wb")
    pk.dump(prot_to_id, f)
    f.close()
   
    
def predict(y_test=False):
    combine(config.raw_fasta_path, config.combined_fasta_path)
    names = cal.cal_main(config.combined_fasta_path, config.num_procs, config.ks, config.freqs_file)
    seq_meta_data_process()    
    batch_data()      
    B_SIZE = 100
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    test_pkls=[]
    test_pkls.append(str(config.freqs_file))

    """examines the model"""
    all_prot_ids = []
    for pkl_f in test_pkls:
        pickle_file =  open(pkl_f, 'rb')
        dataset = pk.load(pickle_file)
        if B_SIZE < len(dataset):
            loader = torch.utils.data.DataLoader(dataset, batch_size =B_SIZE, shuffle=False, drop_last=False)
        else:
            loader = torch.utils.data.DataLoader(dataset, batch_size =B_SIZE, shuffle=False)
        loop = tqdm(loader, leave=True)
        for batch in loop:
            all_prot_ids.append(batch[1])
    
    X = np.concatenate(all_prot_ids, axis=0)
    X = np.array(X) 
    X = np.asarray(X)    
    label_path = '../../../src/datasets/Large-Scale/VFDB/VFDB_num.csv'
    df = pd.read_csv(label_path)
    y = df['VF Cate Code'].values
    y = np.array(y)     
    k_fold = KFold(n_splits=5, shuffle=True, random_state=16) 
    predictor = LogisticRegression(random_state=16, max_iter=1000)     
    y_real = []
    y_proba = [] 
    for i, (train_index, test_index) in enumerate(k_fold.split(X)):
        Xtrain, Xtest = X[train_index], X[test_index]
        ytrain, ytest = y[train_index], y[test_index]
        predictor.fit(Xtrain, ytrain) 
        pred_proba = predictor.predict_proba(Xtest)
        y_pred = predictor.predict(Xtest)
        y_real.append(ytest)
        y_proba.append(pred_proba[:,1])        
        report = classification_report(ytest, y_pred, output_dict=True)
        print(f"Fold {i+1} - Overall Accuracy:", accuracy_score(ytest, y_pred))
        print(f"Macro-Precision:", report['macro avg']['precision'])
        print(f"Macro-Recall:", report['macro avg']['recall'])
        print(f"Macro-F1:", report['macro avg']['f1-score'])
        print(f"Weighted-Precision:", report['weighted avg']['precision'])
        print(f"Weighted-Recall:", report['weighted avg']['recall'])
        print(f"Weighted-F1:", report['weighted avg']['f1-score'])
        print('-'*50)

if __name__ == '__main__':
    config = Config()
    predict()