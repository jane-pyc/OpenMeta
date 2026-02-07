import os
GPU_NUMBER = [0,1,2,3,4,5,6,7]
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(s) for s in GPU_NUMBER])

import time
import datetime
from time import *
from tqdm import trange
from Bio import SeqIO
import subprocess
import logging

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, matthews_corrcoef
import numpy as np
import pickle as pk
from tqdm import tqdm

from torch import nn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc, average_precision_score, roc_curve, classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import pandas as pd
import matplotlib as mpl

import xgboost as xgb
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split

def setup_logger(name, log_file, level=logging.INFO):
    """Function to setup as many loggers as you want"""

    handler = logging.FileHandler(log_file)        
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger


def seq_meta_data_process(fasta_path, new_meta_file_path):
    fasta_path = fasta_path
    print(fasta_path) 
    new_meta_file_path = new_meta_file_path    
    new_data = pd.DataFrame(columns=["contigs", "MGYP_proteins"])
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

    print(new_data.shape)
    new_meta_file_dir = new_meta_file_path.split('/')
    new_data.to_csv(new_meta_file_path, header=None, index=False)

def batch_data(emb_f, contig_to_prots_f, batch_data_path, batch_data_prot_index_dict_path):
    emb_f = emb_f
    contig_to_prots_f = contig_to_prots_f
    batch_data_path = batch_data_path
    batch_data_prot_index_dict_path = batch_data_prot_index_dict_path

    contigs_to_prots_file = open(contig_to_prots_f, "r")
    emb_file = open(emb_f, 'rb')
    esm_embs = pk.load(emb_file)
    embs = []
    
    path = '../../DNABERT2/testdata/DNABert2_Operons.pkl'
    emb_file = open(path, 'rb')
    esm_embs_DNA = pk.load(emb_file)
    all_prot_ids = []
    for key,val in esm_embs_DNA.items():
        all_prot_ids.append(key)
    
    for key,val in esm_embs.items():
        all_prot_ids.append(key)
        embs.append(val)
    emb_file.close()
    embs = np.array(embs, dtype = np.float16)
    embs = embs[0].mean(1)[:4304,:]
    normalized_embs = embs

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
    print(str(count)+" prots processed")

    f = open(batch_data_path, "wb")
    pk.dump(batch,f)
    f.close()
    f = open(batch_data_prot_index_dict_path, "wb")
    pk.dump(prot_to_id, f)
    f.close()
   
def read_annot_file(annot_path):
    f = open(annot_path)
    annot_dict = {}
    for line in f:
        l = line.strip().split("\t")
        annot_dict[int(l[0])] = (l[1],l[2],l[3],l[4])

    return annot_dict

def get_annotations(ids, annot_dict):
    annots = []
    for i in ids:
        if i != 0:
            annots.append(annot_dict[i][0])
    return annots

def same_operon(i, j, annot_dict):
    
    if i == j:
        return False
    if annot_dict[i][2] == "None":
        return False
    elif annot_dict[j][2] == "None":
        return False
    else:
        return annot_dict[i][2] == annot_dict[j][2]
    

def predict(y_test=False):
    fasta_path = '../../../src/datasets/Small-Scale/E-K12/E-K12.fasta'
    new_meta_file_path = './testdata/Gene_operons.csv'
    emb_f = './testdata/NT_Operons.pkl'
    contig_to_prots_f = new_meta_file_path
    batch_data_path = './testdata/NT_Operons_batchdata.pkl'
    batch_data_prot_index_dict_path = './testdata/NT_Operons_protindex.pkl'
    
    seq_meta_data_process(fasta_path,new_meta_file_path)
    batch_data(emb_f,contig_to_prots_f,batch_data_path,batch_data_prot_index_dict_path)
    
    annot_path = '../../../src/datasets/Small-Scale/E-K12/E-K12.annot'
    annot_dict= read_annot_file(annot_path)
    
    B_SIZE = 100
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    test_pkls=[]
    test_pkls.append(str(batch_data_path))

    """examines the model"""
    
    all_prot_ids = []
    for pkl_f in test_pkls:
        pickle_file =  open(pkl_f, 'rb')
        dataset = pk.load(pickle_file)
        if B_SIZE < len(dataset):
            loader = torch.utils.data.DataLoader(dataset, batch_size =B_SIZE, shuffle=False, drop_last=True)
        else:
            loader = torch.utils.data.DataLoader(dataset, batch_size =B_SIZE, shuffle=False)
        loop = tqdm(loader, leave=True)
        for batch in loop:
            inputs_embeds= batch['embeds'].type(torch.FloatTensor) 
            prot_ids = batch['prot_ids'] #
            all_prot_ids.append(prot_ids)
    
    all_prot_ids = np.concatenate(all_prot_ids, axis = 0 )
    ids = all_prot_ids
    multi_head_contacts = inputs_embeds.to(device).cpu().detach().numpy()
    X = []
    y = []
    for i in range(inputs_embeds.shape[0]):
        prot_ids = ids[i]
        annots = get_annotations(prot_ids, annot_dict)
        for ind_a, a in enumerate(prot_ids):
            for ind_b, b in enumerate(prot_ids):
                if a > 0 and b > 0 and ind_b > ind_a and ind_b - ind_a < 2 and ind_b != ind_a :
                    X.append(multi_head_contacts[ind_a,ind_b,:].ravel())
                    if same_operon(a,b,annot_dict):                     
                        y.append(1)
                    else:
                        y.append(0)
    
    X = np.asarray(X)
    y = np.array(y)
    
    k_fold = KFold(n_splits=5, shuffle=True, random_state=16)
    predictor = GaussianNB()
    
    y_real = []
    y_proba = [] 
    e = datetime.datetime.now() 
    output_path = "operon"+ e.strftime("-%d-%m-%Y-%H:%M:%S")
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    if not os.path.exists(output_path+"/figures"):
        os.mkdir(output_path+"/figures")
    FIGURES_DIR = output_path+"/figures/"
    

    for i, (train_index, test_index) in enumerate(k_fold.split(X)):
        Xtrain, Xtest = X[train_index], X[test_index]
        ytrain, ytest = y[train_index], y[test_index]
        predictor.fit(Xtrain, ytrain) 
        pred_proba = predictor.predict_proba(Xtest)
        precision, recall, _ = precision_recall_curve(ytest, pred_proba[:,1])
        y_real.append(ytest)
        y_proba.append(pred_proba[:,1])
        
        y_pred_probs = pred_proba[:,1]
        y_pred = torch.where(torch.from_numpy(y_pred_probs) > 0.5, torch.ones_like(torch.from_numpy(y_pred_probs)), torch.zeros_like(torch.from_numpy(y_pred_probs)))
        accuracy = accuracy_score(ytest, y_pred)
        roc = roc_auc_score(ytest, y_pred)
        f1 = f1_score(ytest, y_pred)
        mcc = matthews_corrcoef(ytest, y_pred)
        mAP = average_precision_score(ytest, pred_proba[:,1])
        print('\n accuracy:', accuracy)
        print('f1:', f1)
        print('roc:', roc)
        print('mcc:', mcc)
        print('mAP: ',mAP)

    y_real = np.concatenate(y_real)
    y_proba = np.concatenate(y_proba)
    precision, recall, _ = precision_recall_curve(y_real, y_proba)
    ypred = predictor.predict(Xtest)
    print(classification_report(ytest, ypred))
    print("\n AC:",accuracy_score(ytest, ypred))
    report_logger = setup_logger('report_logger', './testdata_output/report_log.log')

    y_test = ytest
    accuracy = accuracy_score(y_test, y_pred)    
    print("\nTest Data Accuracy: {}".format(accuracy))
    report_logger.info("Test Data Accuracy: {}".format(accuracy))
    print("\ndata: Operons")
    report_logger.info("data: Operons")

    report = classification_report(y_test, ypred, output_dict=True)
    print(f"Overall Accuracy:", accuracy)
    print(f"Macro-Precision:", report['macro avg']['precision'])
    print(f"Macro-Recall:", report['macro avg']['recall'])
    print(f"Macro-F1:", report['macro avg']['f1-score'])
    print(f"Weighted-Precision:", report['weighted avg']['precision'])
    print(f"Weighted-Recall:", report['weighted avg']['recall'])
    print(f"Weighted-F1:", report['weighted avg']['f1-score'])
    print('-'*50)
    
    report_logger.info("Overall Accuracy: %s", accuracy)
    report_logger.info("Macro-Precision: %s", report['macro avg']['precision'])
    report_logger.info("Macro-Recall: %s", report['macro avg']['recall'])
    report_logger.info("Macro-F1: %s", report['macro avg']['f1-score'])
    report_logger.info("Weighted-Precision: %s", report['weighted avg']['precision'])
    report_logger.info("Weighted-Recall: %s", report['weighted avg']['recall'])
    report_logger.info("Weighted-F1: %s", report['weighted avg']['f1-score'])
    report_logger.info('-'*50)


if __name__ == '__main__':
    # s = time.time()
    predict()
    # print('costs:', time.time() - s)