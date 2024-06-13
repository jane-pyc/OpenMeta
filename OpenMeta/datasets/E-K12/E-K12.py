# usage: python operon_logreg.py -d ../data/ecoli_operon_data/batched_ecoli/ -m ../model/glm.bin --annot ../data/ecoli_operon_data/operon.annot
import os
GPU_NUMBER = [0,1,2,3]
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(s) for s in GPU_NUMBER])
import warnings
warnings.filterwarnings('ignore')
import torch
import sys
from torch import nn
from gLM import *
from glm_utils import *
from transformers import RobertaConfig
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
import random
import numpy as np
import argparse
import pathlib
import matplotlib.pyplot as plt
import datetime
import logging
import scipy.stats
import scipy.special
import pickle as pk
from sklearn.metrics import precision_recall_curve, auc, average_precision_score, roc_curve, classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, matthews_corrcoef
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
import pandas as pd
import matplotlib as mpl

random.seed(1)
torch.cuda.empty_cache()

def read_annot_file(annot_path):
    f = open(annot_path)
    annot_dict = {}
    for line in f:
        l = line.strip().split("\t")
        annot_dict[int(l[0])] = (l[1],l[2],l[3],l[4])

    return annot_dict

def get_operons(ids, annot_dict):
    operons = []
    for i in ids:
        if i != 0:
            operons.append(annot_dict[i][2])
    return operons

def get_annotations(ids, annot_dict):
    annots = []
    for i in ids:
        if i != 0:
            annots.append(annot_dict[i][0])
    return annots

def get_descriptions(ids, annot_dict):
    descs = []
    for i in ids:
        if i != 0:
            descs.append(annot_dict[i][1])
    return descs

def same_operon(i, j, annot_dict):
    
    if i == j:
        return False
    if annot_dict[i][2] == "None":
        return False
    elif annot_dict[j][2] == "None":
        return False
    else:
        return annot_dict[i][2] == annot_dict[j][2]

def draw_operon(ind,ids,annot_dict,all_attentions,predictor,max_cor_mat_inds,ax):
    prot_ids = ids[ind]
    annots = get_annotations(prot_ids, annot_dict)
    max_head =  all_attentions[ind,:len(annots),:len(annots),NHEADS*max_cor_mat_inds[0]+max_cor_mat_inds[1]].squeeze()
    contacts = all_attentions[ind]
    contacts  = contacts.reshape(900,190)
    output_mat = np.zeros((3,len(annots)-1))
    
    for ind_a, a in enumerate(prot_ids):
        for ind_b, b in enumerate(prot_ids):
            if a > 0 and b > 0 and ind_b > ind_a and ind_b - ind_a < 2 and ind_b != ind_a :
                pred = predictor.predict(np.expand_dims(contacts[30*ind_a+ind_b],axis =0))
                output_mat[1][ind_a] = max_head[ind_a][ind_b]
                if same_operon(a,b,annot_dict):
                    output_mat[2][ind_a] = 1 
                if pred == 1 and same_operon(a,b,annot_dict):
                    output_mat[0][ind_a] = 1                     
                elif pred == 1 and not same_operon(a,b,annot_dict):
                    output_mat[0][ind_a] = -1   
    ylabels = ["LogReg", "L"+str(max_cor_mat_inds[0]+1)+" -H"+str(max_cor_mat_inds[1]+1)+" raw", "Operons"]
    ax.pcolor(output_mat,cmap='RdBu',vmax = 1.5, vmin=-1.5)
    ax.set_xticks(np.arange(output_mat.shape[1]+1), minor=False)
    ax.set_yticks(np.arange(output_mat.shape[0]) + 0.5, minor=False)
    ax.set_xticklabels(annots,rotation=90, minor=False)
    ax.set_yticklabels(ylabels, minor=False)
    ax.set_aspect('equal')
    return None

def visualize_operons(all_contacts,ids,annot_dict,logging):
    X = []
    y = []
    for i, multi_head_contacts in enumerate(all_contacts):
        prot_ids = ids[i]
        annots = get_annotations(prot_ids, annot_dict)
        prot_ids = ids[i]
        
        for ind_a, a in enumerate(prot_ids):
            for ind_b, b in enumerate(prot_ids):
                if a > 0 and b > 0 and ind_b > ind_a and ind_b - ind_a < 2 and ind_b != ind_a :
                    X.append(multi_head_contacts[ind_a,ind_b,:].ravel())
                    if same_operon(a,b,annot_dict):                            
                        y.append(1)
                    else:
                        y.append(0)

    k_fold = KFold(n_splits=5, shuffle=True, random_state=16)
    predictor = RandomForestClassifier()
    y_real = []
    y_proba = []
    X = np.array(X)
    y = np.array(y)
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
        
        logging.info("accuracy: "+str(accuracy))
        logging.info("f1: "+str(f1))
        logging.info("roc: "+str(roc))
        logging.info("mcc: "+str(mcc))
        logging.info("mAP: "+str(mAP))
    y_real = np.concatenate(y_real)
    y_proba = np.concatenate(y_proba)
    precision, recall, _ = precision_recall_curve(y_real, y_proba)
    ypred = predictor.predict(Xtest)
    report = classification_report(ytest, ypred, output_dict=True)
    print(classification_report(ytest, ypred))
    
    print("AC:",accuracy_score(ytest, ypred))    
    logging.info("AC: "+str(accuracy_score(ytest, ypred)))
    
    print(f"Overall Accuracy:", accuracy_score(ytest, ypred))
    print(f"Macro-Precision:", report['macro avg']['precision'])
    print(f"Macro-Recall:", report['macro avg']['recall'])
    print(f"Macro-F1:", report['macro avg']['f1-score'])
    print(f"Weighted-Precision:", report['weighted avg']['precision'])
    print(f"Weighted-Recall:", report['weighted avg']['recall'])
    print(f"Weighted-F1:", report['weighted avg']['f1-score'])
    print('-'*50)
    
    logging.info("Overall Accuracy: %s", accuracy_score(ytest, ypred))
    logging.info("Macro-Precision: %s", report['macro avg']['precision'])
    logging.info("Macro-Recall: %s", report['macro avg']['recall'])
    logging.info("Macro-F1: %s", report['macro avg']['f1-score'])
    logging.info("Weighted-Precision: %s", report['weighted avg']['precision'])
    logging.info("Weighted-Recall: %s", report['weighted avg']['recall'])
    logging.info("Weighted-F1: %s", report['weighted avg']['f1-score'])
    logging.info('-'*50)

def examine_model(logging, data_dir, model, device, annot_dict):
    f_list = os.listdir(data_dir)
    test_pkls=[]
    for pkl_f in f_list:
        test_pkls.append(str(os.path.join(data_dir,pkl_f)))
    torch.cuda.empty_cache()
    logging.info("testing model...")
    scaler = None
    input_embs = []
    hidden_embs = []
    all_contacts = []
    all_prot_ids = []
    predicted_embeds_masked = []
    all_probs = []
    if HALF:
        logging.info("testing using a mixed precision model")
        scaler = torch.cuda.amp.GradScaler()
    for pkl_f in test_pkls:
        pickle_file =  open(pkl_f, 'rb')
        dataset = pk.load(pickle_file)
        id_to_label = {}
        for seq in dataset:
            labels =seq['label_embeds']
            prot_ids = seq['prot_ids']
            for ind, i in enumerate(prot_ids):
                if i !=0:
                    id_to_label[i] = labels[ind]       
        if B_SIZE < len(dataset):
            loader = torch.utils.data.DataLoader(dataset, batch_size =B_SIZE, shuffle=False, drop_last=True)
        else:
            loader = torch.utils.data.DataLoader(dataset, batch_size =B_SIZE, shuffle=False)
        loop = tqdm(loader, leave=True)
        for batch in loop:
            # pull all tensor batches required for testing
            inputs_embeds= batch['embeds'].type(torch.FloatTensor)
            attention_mask = batch['attention_mask'].type(torch.FloatTensor)
            mask = torch.zeros(attention_mask.shape) #nothing is masked
            masked_tokens = (mask==1) & (attention_mask != 0)
            masked_tokens = torch.unsqueeze(masked_tokens, -1)
            masked_tokens = masked_tokens.to(device)
            inputs_embeds = inputs_embeds.to(device)
            inputs_embeds  = torch.where(masked_tokens, -1.0, inputs_embeds)
            attention_mask = attention_mask.to(device)
            labels = batch['label_embeds'].type(torch.FloatTensor)
            labels = labels.to(device)
            input_embs.append(inputs_embeds.cpu().detach().numpy())
            if scaler is not None:
                with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                    outputs = model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels = labels, masked_tokens = masked_tokens, output_attentions = False)
                    last_hidden_states = outputs.last_hidden_state
                    hidden_embs.append(last_hidden_states.cpu().detach().numpy())
                    prot_ids = batch['prot_ids']
                    all_contacts.append(outputs.contacts.cpu().detach().numpy())
                    all_prot_ids.append(prot_ids)
                    logits_all_preds = outputs.logits_all_preds
                    all_preds = logits_all_preds[masked_tokens.squeeze(-1)]
                    predicted_probs = outputs.probs
                    raw_probs = predicted_probs.view(-1,4)
                    softmax = nn.Softmax(dim=1)
                    probs = softmax(raw_probs)
                    predicted_embeds_masked.append(all_preds.cpu().detach().numpy())
                    all_probs.append(probs.cpu().detach().numpy())
    input_embs = np.concatenate(input_embs, axis = 0)
    hidden_embs = np.concatenate(hidden_embs, axis=0)
    all_prot_ids = np.concatenate(all_prot_ids, axis = 0 )
    all_contacts = np.concatenate(all_contacts, axis= 0)
    visualize_operons(all_contacts,all_prot_ids,annot_dict,logging)
    return None
parser = argparse.ArgumentParser(description = "outputs glm embeddings")
parser.add_argument('-d','--data_dir', type=pathlib.Path, help='batched data directory', default='../../../src/datasets/Small-Scale/E-K12/E-K12.pkl')
parser.add_argument('-id', '--id_path',  help='path to prot_index_dict.pkl file', default = None)
parser.add_argument('-m','--model_path', help="path to pretrained model, glm.bin", default = './model/pytorch_model.bin')
parser.add_argument('-b','--batch_size', type=int, help='batch_size', default = 100)
parser.add_argument('-o', '--output_path', type=str, help='inference output directory', default = None)
parser.add_argument('--attention',action='store_true', help='output attention matrices ', default = False)
parser.add_argument('--hidden_size', type=int, help='hidden size', default = 1280)
parser.add_argument('-n', '--ngpus', type=int, help='number of GPUs to use',  default = 1)
parser.add_argument('--annot_path',help='path to operon annotation file', default='../../../src/datasets/Small-Scale/E-K12/E-K12.annot')
parser.add_argument('-a', '--all_results',action='store_true', help='output all results including plm_embs/glm_embs/prot_ids/outputs/output_probabilitess', default = True)
args = parser.parse_args()
if args.data_dir is None :
    parser.error('--data_dir must be specified')
if args.model_path is None :
    parser.error('--model must be specified')
if args.annot_path is None :
    parser.error('--annot_path must be specified')
data_dir = args.data_dir
ngpus = args.ngpus
model_path = args.model_path
num_pred = 4
max_seq_length = 30 
output_path = args.output_path
pos_emb = "relative_key_query"
pred_probs = True
id_path = args.id_path
annot_path = args.annot_path
HIDDEN_SIZE = args.hidden_size
B_SIZE = args.batch_size
HALF = True
EMB_DIM = 1281
NUM_PC_LABEL = 100
ATTENTION = args.attention
ALL_RESULTS = args.all_results
NHEADS = 10
NLAYERS = 19
e = datetime.datetime.now()
if output_path == None:
    output_path = "./operons_figure/"+ e.strftime("-%d-%m-%Y-%H:%M:%S")
if not os.path.exists(output_path):
    os.mkdir(output_path)
logfile_path = output_path+"/info.log"
if not os.path.exists(output_path+"/figures"):
    os.mkdir(output_path+"/figures")
FIGURES_DIR = output_path+"/figures/"

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO, handlers=[
        logging.FileHandler(logfile_path),
        logging.StreamHandler()])
logging.info("output folder: " +output_path)
logging.info("log file is located here: " +logfile_path)
string_of_command = f"{' '.join(sys.argv)}"
logging.info("command: " + string_of_command)
config = RobertaConfig(
    max_position_embedding = max_seq_length,
    hidden_size = HIDDEN_SIZE,
    num_attention_heads = NHEADS,
    type_vocab_size = 1,
    tie_word_embeddings = False,
    num_hidden_layers = NLAYERS,
    num_pc = NUM_PC_LABEL, 
    num_pred = num_pred,
    predict_probs = pred_probs,
    emb_dim = EMB_DIM,
    output_attentions=True,
    output_hidden_states=True,
    position_embedding_type = pos_emb,
)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model =  gLM(config)
model.load_state_dict(torch.load(model_path, map_location=device),strict=False)
model.eval()
if ngpus>1:
    model = torch.nn.DataParallel(model)
model.to(device)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
logging.info("batch_size: "+str(B_SIZE))
if id_path != None:
    id_dict = pk.load(open(id_path, "rb"))
else:
    id_dict = None
annot_dict= read_annot_file(annot_path)
with torch.no_grad():
    examine_model(logging, data_dir, model, device, annot_dict)