import os
GPU_NUMBER = [0,4,5,6,7]
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(s) for s in GPU_NUMBER])
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from collections import Counter
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import pickle as pk
from sklearn.preprocessing import label_binarize
from tqdm import tqdm  
from sklearn.metrics import classification_report,precision_recall_curve,average_precision_score
from sklearn.metrics import accuracy_score,confusion_matrix
import logging

torch.manual_seed(12345)
class MulticlassClassification(nn.Module):
    def __init__(self, num_feature, num_class):
        super(MulticlassClassification, self).__init__()
        self.layer_1 = nn.Linear(num_feature, 2048)
        self.layer_2 = nn.Linear(2048, 1024)
        self.layer_3 = nn.Linear(1024, 512)
        self.layer_out = nn.Linear(512, num_class) 
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.batchnorm1 = nn.BatchNorm1d(2048)
        self.batchnorm2 = nn.BatchNorm1d(1024)
        self.batchnorm3 = nn.BatchNorm1d(512)
        
        
    def forward(self, x):
        x = self.layer_1(x)
        x = self.batchnorm1(x)
        x = self.relu(x)
        
        x = self.layer_2(x)
        x = self.batchnorm2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.layer_3(x)
        x = self.batchnorm3(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.layer_out(x)        
        return x

class ClassifierDataset(Dataset):
    
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data
        
    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]
        
    def __len__ (self):
        return len(self.X_data)

def setup_logger(name, log_file, level=logging.INFO):
    """Function to setup as many loggers as you want"""
    handler = logging.FileHandler(log_file)        
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    return logger

def extract_ec_data_from_results_pkls(result_pkl_path, type):
    result_pkl_f = open(result_pkl_path, "rb")
    results = pk.load(result_pkl_f)
    prot_ids = results['all_prot_ids']
    input_embs = results['plm_embs']
    input_embs_new=input_embs 
    hidden_embs = results['glm_embs']
    hidden_embs_new=hidden_embs

    if type == "plmglm":
        label_hidden_concat = np.concatenate((input_embs_new[:,:-1], hidden_embs_new), axis=1)
        return label_hidden_concat, prot_ids
    elif type == "plm":
        return input_embs_new[:,:-1], prot_ids
        
def extract_ec_data_from_results_pkls_sequential(results_pkls, type):
    embs = []
    prot_ids = []
    for result_pkl in tqdm(results_pkls, total=len(results_pkls)):
        h,prot_id = extract_ec_data_from_results_pkls(result_pkl, type)
        embs.extend(h)
        prot_ids.extend(prot_id)
    embs = np.array(embs)
    prot_ids = np.array(prot_ids) 
    return  embs, prot_ids

def save_data(results_dir, type, file_name):
    f_list = os.listdir(results_dir)
    results_pkls = []    
    for pkl_f in f_list:
        if pkl_f.startswith('train.pkl.results'):
            results_pkls.append(str(os.path.join(results_dir,pkl_f)))
    first_100 = results_pkls[:10]
    embs, prot_ids= extract_ec_data_from_results_pkls_sequential(first_100, type)    
    pkl_filename = f"LARGE.{type}_{file_name}.pkl" 
    pkl_filepath = os.path.join(results_dir, pkl_filename) 
    with open(pkl_filepath, "wb") as pkl_file:
        pk.dump([embs, prot_ids], pkl_file)       
    return None

def multi_acc(y_pred, y_test):
    y_pred_softmax = torch.log_softmax(y_pred, dim = 1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim = 1)    
    correct_pred = (y_pred_tags == y_test).float()
    acc = correct_pred.sum() / len(correct_pred)
    acc = torch.round(acc * 100)
    return acc

def split_data(data_path, type, file_name, cate):
    if file_name.find('testdata')!=-1:  
        dat = pk.load(open(os.path.join(data_path, f'LARGE.{type}_{file_name}.pkl'),"rb"))
        X_test = dat[0]
        y_test = dat[1]
    elif file_name == 'traning_data':
        dat = pk.load(open(os.path.join(data_path, f'LARGE.{type}_{file_name}.pkl'),"rb"))
        X_train = dat[0]
        y_train = dat[1]
    
    if file_name.find('testdata')!=-1:
        if cate == 'GeneName':
            csv_path = '../../../src/datasets/Fine-Grained/NCRD-F/NCRD_GeneName_num.csv'
            df = pd.read_csv(csv_path)
            df['Seq ID'] = df['Seq ID'].astype(str)
            y_train_output = df.set_index('Seq ID').loc[y_train[:len(df)]]['Gene Name Code'].values  
        elif cate == 'GeneFamily':
            csv_path = '../../../src/datasets/Fine-Grained/NCRD-N/NCRD_GeneFamily_num.csv'
            df = pd.read_csv(csv_path)
            df['Seq ID'] = df['Seq ID'].astype(str)
            y_train_output = df.set_index('Seq ID').loc[y_train[:len(df)]]['Gene Family Code'].values
        elif cate == 'Resistance':
            csv_path = '../../../src/datasets/Fine-Grained/NCRD-C/NCRD_Resistance_num.csv'
            df = pd.read_csv(csv_path)
            df['Seq ID'] = df['Seq ID'].astype(str)       
            y_train_output = df.set_index('Seq ID').loc[y_train[:len(df)]]['Resistance Code'].values 
        elif cate == 'Mechanism':
            csv_path = '../../../src/datasets/Fine-Grained/NCRD-R/NCRD_Mechanism_num.csv'
            df = pd.read_csv(csv_path)
            df['Seq ID'] = df['Seq ID'].astype(str)       
            y_train_output = df.set_index('Seq ID').loc[y_train[:len(df)]]['Mechanism Code'].values           

    if file_name.find('testdata')!=-1:    
        test_split_path = "LARGE_test_split." + type + '_' + file_name + ".pkl"
        test_dat_f = open(os.path.join(data_path, test_split_path), "wb")
        pk.dump([X_test[:y_test_output.size], y_test_output], test_dat_f)
        test_dat_f.close()
    elif file_name == 'traning_data':
        train_split_path = "LARGE_train_split." + type + '_' + file_name + ".pkl"
        train_dat_f = open(os.path.join(data_path, train_split_path), "wb")
        pk.dump([X_train[:y_train_output.size], y_train_output], train_dat_f)
        train_dat_f.close()


def train(data_path, type, file_name, cate, test_data, all_probabilities, report_logger):
    X_test, y_test = pk.load(open(os.path.join(data_path, f'LARGE_test_split.{type}_{file_name}.pkl'),"rb"))         
    pattern = r'testdata\d+'  
    replacement = 'traning_data' 
    X_train, y_train = pk.load(open(os.path.join(data_path, f'LARGE_test_split.{type}_{file_name}.pkl'),"rb"))     
    if type == "plmglm":
        nfeatures = 2560
    else:
        nfeatures =1280
    print("successfully loaded data")

    train_dataset = ClassifierDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long())
    test_dataset = ClassifierDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).long())    
    entire_y = np.concatenate((y_train, y_test), axis=0) 
    class_count = [0] * (max([int(i) for i in entire_y]) + 1) 
    y_counter = Counter(entire_y)     
    for key, val in y_counter.items():
        class_count[int(key)] = val
    class_weights = 1./torch.tensor(class_count, dtype=torch.float)
    train_weights = class_weights[torch.tensor([int(i) for i in y_train])].numpy()
    weighted_sampler = WeightedRandomSampler(weights=train_weights, num_samples=len(y_train), replacement=True)

    EPOCHS = 650
    BATCH_SIZE = 5000
    LEARNING_RATE = 0.0001
    NUM_FEATURES = nfeatures
    NUM_CLASSES = len(class_count)
    print("NUM_CLASSES: ", NUM_CLASSES)
    train_loader = DataLoader(dataset=train_dataset,
                            batch_size=BATCH_SIZE,
                            sampler=weighted_sampler
    )
    val_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = MulticlassClassification(num_feature = NUM_FEATURES, num_class=NUM_CLASSES)
    model.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    accuracy_stats = {
        'train': [],
        "val": []
    }
    loss_stats = {
        'train': [],
        "val": []
    }
    
    for e in range(1, EPOCHS+1):
        # TRAINING
        train_epoch_loss = 0
        train_epoch_acc = 0
        model.train()
        loop = tqdm(train_loader, leave=True)
        for X_train_batch, y_train_batch in loop:
            X_train_batch, y_train_batch = X_train_batch.to(device), y_train_batch.to(device)
            optimizer.zero_grad()
            y_train_pred = model(X_train_batch)
            train_loss = criterion(y_train_pred, y_train_batch)
            train_acc = multi_acc(y_train_pred, y_train_batch)
            train_loss.backward()
            optimizer.step()
            train_epoch_loss += train_loss.item()
            train_epoch_acc += train_acc.item()
            loop.set_description(f'Epoch {e}')
            loop.set_postfix(train_loss=train_loss.item(),accuracy=train_acc.item())
            
        # VALIDATION    
        with torch.no_grad():
            val_epoch_loss = 0
            val_epoch_acc = 0
            model.eval()
            loop = tqdm(val_loader, leave=True)
            for X_val_batch, y_val_batch in loop:
                X_val_batch, y_val_batch = X_val_batch.to(device), y_val_batch.to(device)
                y_val_pred = model(X_val_batch)
                val_loss = criterion(y_val_pred, y_val_batch)
                val_acc = multi_acc(y_val_pred, y_val_batch)
                val_epoch_loss += val_loss.item()
                val_epoch_acc += val_acc.item()
                loop.set_description(f'Epoch {e}')
                loop.set_postfix(train_loss=val_loss.item(),accuracy=val_acc.item())
        loss_stats['train'].append(train_epoch_loss/len(train_loader))
        loss_stats['val'].append(val_epoch_loss/len(val_loader))
        accuracy_stats['train'].append(train_epoch_acc/len(train_loader))
        accuracy_stats['val'].append(val_epoch_acc/len(val_loader))          
        print(f'Epoch {e+0:03}: | Train Loss: {train_epoch_loss/len(train_loader):.5f} | Val Loss: {val_epoch_loss/len(val_loader):.5f} | Train Acc: {train_epoch_acc/len(train_loader):.3f}| Val Acc: {val_epoch_acc/len(val_loader):.3f}')
  
    model_name = data_path.split('output')[1].split('/results')[0][1:] 
    figure_path = os.path.join('output_figure', model_name)
    figure_path = os.path.abspath(figure_path)
    if not os.path.exists(figure_path):
        os.makedirs(figure_path)
    y_pred_list = []
    y_pred_proba = []
    with torch.no_grad():
        model.eval()
        for X_batch, _ in tqdm(test_loader):
            X_batch = X_batch.to(device)
            y_test_pred = model(X_batch)
            _, y_pred_tags = torch.max(y_test_pred, dim = 1)
            y_pred_list.append(y_pred_tags.cpu().numpy())
            y_pred_proba.append(y_test_pred.cpu().numpy())
    y_pred_list= np.concatenate(y_pred_list,axis =0)
    y_pred_list = [a.squeeze().tolist() for a in y_pred_list]
    y_pred_proba = np.concatenate(y_pred_proba,axis = 0)
    report= classification_report(y_test, y_pred_list, output_dict=True)    
    precision = dict()
    recall = dict()
    average_precision = dict()
    y_bin = label_binarize(y_test, classes=list(range(max(y_train)+1)))    
    precision["micro"], recall["micro"], _ = precision_recall_curve(
        y_bin.ravel(), np.array(y_pred_proba).ravel()
    )
    average_precision["micro"] = average_precision_score(y_bin, y_pred_proba, average="micro")
    y_pred = np.argmax(y_pred_proba, axis=1)
    accuracy = accuracy_score(y_test, y_pred)
    print("cate: {}".format(cate))
    print("NUM_CLASSES: {}".format(NUM_FEATURES))
    print("NUM_CLASSES: {}".format(NUM_CLASSES))
    print("test-data: {}".format(test_data))
    report_logger.info("cate: {}".format(cate))
    report_logger.info("NUM_FEATURES: {}".format(NUM_FEATURES))
    report_logger.info("NUM_CLASSES: {}".format(NUM_CLASSES))
    report_logger.info("test-data: {}".format(test_data))    
    cm = confusion_matrix(y_test, y_pred)
    TP = np.diag(cm) 
    FP = cm.sum(axis=0) - TP  
    FN = cm.sum(axis=1) - TP
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1_score = 2 * (precision * recall) / (precision + recall)
    macro_precision = np.nanmean(precision)
    macro_recall = np.nanmean(recall)
    macro_f1_score = np.nanmean(f1_score)
    weights = cm.sum(axis=1)
    weighted_precision = np.average(precision, weights=weights)
    weighted_recall = np.average(recall, weights=weights)
    weighted_f1_score = np.average(f1_score, weights=weights)
    
    TN = cm.sum() - (FP + FN + TP)
    fpr = FP / (FP + TN)
    fnr = FN / (TP + FN)
    tnr = TN / (TN + FP)
    tpr = TP / (TP + FN)

    macro_fpr = np.nanmean(fpr)
    macro_fnr = np.nanmean(fnr)
    macro_tnr = np.nanmean(tnr)
    macro_tpr = np.nanmean(tpr)    
    weighted_fpr = np.average(fpr, weights=weights)
    weighted_fnr = np.average(fnr, weights=weights)
    weighted_tnr = np.average(tnr, weights=weights)
    weighted_tpr = np.average(tpr, weights=weights)

    print(f"Macro-FPR: {macro_fpr}")
    print(f"Macro-FNR: {macro_fnr}")
    print(f"Weighted-FPR: {weighted_fpr}")
    print(f"Weighted-FNR: {weighted_fnr}")
    print(f"Macro-TNR: {macro_tnr}")
    print(f"Macro-TPR: {macro_tpr}")
    print(f"Weighted-TNR: {weighted_tnr}")
    print(f"Weighted-TPR: {weighted_tpr}")
    print('-'*50)
    
    report_logger.info(f"Macro-FPR: {macro_fpr}")
    report_logger.info(f"Macro-FNR: {macro_fnr}")
    report_logger.info(f"Weighted-FPR: {weighted_fpr}")
    report_logger.info(f"Weighted-FNR: {weighted_fnr}")
    report_logger.info(f"Macro-TNR: {macro_tnr}")
    report_logger.info(f"Macro-TPR: {macro_tpr}")
    report_logger.info(f"Weighted-TNR: {weighted_tnr}")
    report_logger.info(f"Weighted-TPR: {weighted_tpr}")
    report_logger.info('-'*50)
    
    
    print("Test Data Accuracy: {}".format(accuracy))
    report_logger.info("Test Data Accuracy: {}".format(accuracy))  

    report = classification_report(y_test, y_pred, output_dict=True)
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

    results_df = pd.DataFrame(columns=['Class Name', 'Total Count', 'Correctly Classified'])
    report_dict = classification_report(y_test, y_pred, output_dict=True)        
    for class_name, metrics in report_dict.items():
        if class_name not in ['accuracy', 'macro avg', 'weighted avg']:
            total_count = metrics['support']
            correctly_classified = int(metrics['precision'] * total_count)
            results_df = results_df.append({'Class Name': class_name, 
                                            'Total Count': total_count, 
                                            'Correctly Classified': correctly_classified}, 
                                        ignore_index=True)    
    report_dir = os.path.join(figure_path, 'classification_report')
    if not os.path.exists(report_dir):
        os.makedirs(report_dir)
    result_path = os.path.join(report_dir, f'LARGE_{type}_{cate}_classification_results.csv')
    print(f"Result path: {result_path}")
    report_logger.info("Result path: %s", result_path)
    print('-'*70)    
    results_df.to_csv(result_path, index=False)  

    if cate == 'GeneName':
        original_df = pd.read_csv('../../../src/datasets/Fine-Grained/NCRD-F/NCRD_GeneName_code.csv') 
        id_to_name = dict(zip(original_df['Seq ID'], original_df['Seq Name']))
        encoded_df = pd.read_csv('../../../src/datasets/Fine-Grained/NCRD-F/NCRD_GeneName_num.csv')
        code_to_name = dict(zip(encoded_df['Gene Name Code'], original_df['Gene Name']))
        columns_plm_arg = encoded_df['Gene Name Code']
        plm_arg_output_file = '../../../src/datasets/Fine-Grained/NCRD-F/NCRD_GeneName_predicted_results.csv'
         
    elif cate == 'GeneFamily':
        original_df = pd.read_csv('../../../src/datasets/Fine-Grained/NCRD-N/NCRD_GeneFamily_code.csv')
        id_to_name = dict(zip(original_df['Seq ID'], original_df['Seq Name']))
        encoded_df = pd.read_csv('../../../src/datasets/Fine-Grained/NCRD-N/NCRD_GeneFamily_num.csv')
        code_to_name = dict(zip(encoded_df['Gene Family Code'], original_df['Gene Family']))
        columns_plm_arg = encoded_df['Gene Family Code']
        plm_arg_output_file = '../../../src/datasets/Fine-Grained/NCRD-N/NCRD_GeneFamily_predicted_results.csv'
    
    elif cate == 'Resistance':
        original_df = pd.read_csv('../../../src/datasets/Fine-Grained/NCRD-C/NCRD_Resistance_code.csv')
        id_to_name = dict(zip(original_df['Seq ID'], original_df['Seq Name']))
        encoded_df = pd.read_csv('../../../src/datasets/Fine-Grained/NCRD-C/NCRD_Resistance_num.csv')
        code_to_name = dict(zip(encoded_df['Resistance Code'], original_df['Resistance']))
        columns_plm_arg = encoded_df['Resistance Code']
        plm_arg_output_file = '../../../src/datasets/Fine-Grained/NCRD-C/NCRD_Resistance_predicted_results.csv'
    
    elif cate == 'Mechanism':
        original_df = pd.read_csv('../../../src/datasets/Fine-Grained/NCRD-R/NCRD_Mechanism_code.csv')
        id_to_name = dict(zip(original_df['Seq ID'], original_df['Seq Name']))
        encoded_df = pd.read_csv('../../../src/datasets/Fine-Grained/NCRD-R/NCRD_Mechanism_code.csv')
        code_to_name = dict(zip(encoded_df['Mechanism Code'], original_df['Mechanism']))
        columns_plm_arg = encoded_df['Mechanism Code']
        plm_arg_output_file = '../../../src/datasets/Fine-Grained/NCRD-R/NCRD_Mechanism_predicted_results.csv'
    
    from scipy.special import softmax
    def create_results_df(y_pred_proba, seq_id, columns_plm_arg, code_to_name, all_probabilities):
        all_probabilities = True
        y_pred_proba = softmax(y_pred_proba, axis=1)  
        column_names = columns_plm_arg.unique().tolist()
        print("y_pred_proba.shape[1] == len(column_names): ", y_pred_proba.shape[1] == len(column_names))
        print("len(column_names): ", len(column_names))
        results_df = pd.DataFrame(y_pred_proba[:,1:], columns=column_names)
        results_df = results_df.rename(columns=code_to_name)
        seq_id = [id_to_name.get(i+1, 'unknown') for i in range(len(y_pred_list))]
        results_df.insert(0, 'seq_id', seq_id)
        pred_codes = np.argmax(y_pred_proba[:, :-1], axis=1)
        pred_names = [code_to_name.get(columns_plm_arg.unique()[code], 'unknown') for code in pred_codes]
        results_df.insert(1, 'pred', pred_names)
        if not all_probabilities:
            results_df = results_df[['seq_id', 'pred']]
        return results_df
    seq_id = ['seq_id_{}'.format(i) for i in range(len(y_pred_list))]
    y_pred_proba = softmax(y_pred_proba, axis=1)  
    column_names = columns_plm_arg.unique().tolist()
    print("y_pred_proba.shape[1] == len(column_names): ", y_pred_proba.shape[1] == len(column_names))
    print("len(column_names): ", len(column_names))
    results_df = pd.DataFrame(y_pred_proba, columns=column_names)
    results_df = results_df.rename(columns=code_to_name)  
    valid_records = []
    for i in range(len(y_pred_list)):
        seq_name = id_to_name.get(i+1, 'unknown')
        if seq_name != 'sp':
            valid_records.append({
                "seq_id": seq_name,
                "pred": code_to_name.get(columns_plm_arg.unique()[np.argmax(y_pred_proba[i])], 'unknown'),
                "max_proba": np.max(y_pred_proba[i])
            })
    valid_results_df = pd.DataFrame(valid_records)
    valid_results_df.to_csv(plm_arg_output_file, index=False)
    results_df = pd.read_csv(result_path)
    results_df['Class Name'] = results_df['Class Name'].astype(int).map(code_to_name)
    result_path_updated = f'{figure_path}/classification_report/LARGE_{type}_{cate}_{test_data}_classification_results_updated.csv'
    results_df.to_csv(result_path_updated, index=False)   
    results_df = pd.read_csv(result_path_updated)
    results_df['Correct Classification Percentage'] = (results_df['Correctly Classified'] / results_df['Total Count']) * 100
    results_df.to_csv(result_path_updated, index=False)

import argparse
parser = argparse.ArgumentParser(description='LARGe predict')
parser.add_argument('--cate', type=str, default='GeneFamily', help='Cate of GeneName, GeneFamily or Resistance Category')
parser.add_argument('--test_data', type=str, default='testdata9', help='Path to the input file')
parser.add_argument('--all_probabilities', type=str, default='False', help='whether or not to output all probabilities')
args = parser.parse_args()

type = 'plmglm'
cate = args.cate
test_data=args.test_data
all_probabilities=args.all_probabilities

def recursive_search(folder_path, type, file_name, report_logger):    
    for root, dirs, files in os.walk(folder_path):
        for dir in dirs:
            if dir == "results":
                results_folder = os.path.join(root, dir)
                print("\n save_data results_folder is:", results_folder)    
                save_data(results_folder, type, file_name)
                split_data(results_folder, type, file_name, cate)
                train(results_folder, type, file_name, cate, test_data, all_probabilities, report_logger)               

file_name=test_data
folder_path = f"./output_Triplet_{file_name}/"
report_logger = setup_logger('report_logger', f'./output_Triplet_{file_name}/report_log.log')
recursive_search(folder_path, type, file_name, report_logger)