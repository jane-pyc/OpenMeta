import os
GPU_NUMBER = [1,2,5,6,7]
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(s) for s in GPU_NUMBER])
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import torch
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, matthews_corrcoef
import numpy as np
import pickle as pk
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict
from collections import Counter
import sys
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from multiprocessing import Pool
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split, KFold
import pdb
import logging
from torch import nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve, auc, average_precision_score, roc_curve, classification_report, confusion_matrix
import matplotlib as mpl
import xgboost as xgb
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

torch.manual_seed(12345)
class MulticlassClassification(nn.Module):
    def __init__(self, num_feature, num_class):
        super(MulticlassClassification, self).__init__()
        self.layer_1 = nn.Linear(num_feature, 2048)
        self.layer_2 = nn.Linear(2048, 1024)
        self.layer_3 = nn.Linear(1024, 514)
        self.layer_out = nn.Linear(514, num_class) 
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)
        self.batchnorm1 = nn.BatchNorm1d(2048)
        self.batchnorm2 = nn.BatchNorm1d(1024)
        self.batchnorm3 = nn.BatchNorm1d(514)
        
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


def multi_acc(y_pred, y_test):
    y_pred_softmax = torch.log_softmax(y_pred, dim = 1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim = 1)    
    correct_pred = (y_pred_tags == y_test).float()
    acc = correct_pred.sum() / len(correct_pred)
    acc = torch.round(acc * 100)
    return acc

def split_data(data_path, type, filter_num):  
    data = pk.load(open(data_path,"rb"))
    keys=[]
    values=[]
    for key, value in data.items():
            keys.append(key.split('_')[-1])
            values.append(value.cpu())
    keys_stack = np.stack(keys)
    values_stack = np.stack(values)
    X = values_stack[:-8]
    y = keys_stack[:-8]
    
    csv_path = '../../../src/datasets/Small-Scale/PATRIC/Patric_num.csv'
    df = pd.read_csv(csv_path)
    df = df.drop_duplicates(subset='genome_id', keep='first') 
    df['genome_id'] = df['genome_id'].astype(str)
    y_output = df.set_index('genome_id').loc[y]['genome_cate_code'].values  
   
    y = y_output    
    ec_counter = Counter(y)
    y_filter = [ec for ec, c in ec_counter.items() if c < filter_num ]
    y_new = np.delete(y,np.where(np.isin(y,y_filter))[0],axis =0)
    X_new = np.delete(X,np.where(np.isin(y,y_filter))[0],axis =0)

    X_trainval, X_test, y_trainval, y_test = train_test_split(X_new, y_new, test_size=0.2, stratify=y_new, random_state=69)
    X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.1, stratify=y_trainval, random_state=21)
    
    train_split_path = "./Patric_"+type+".pkl"
    train_dat_f = open(train_split_path,"wb")
    pk.dump([X_train, y_train, X_val, y_val,X_test, y_test],train_dat_f )
    train_dat_f.close()   

def train_ML(data_path):
    data = pk.load(open(data_path,"rb"))
    keys=[]
    values=[]
    for key, value in data.items():
            keys.append(key.split('_')[-1])
            values.append(value)
    keys_stack = np.stack(keys)
    values_stack = np.stack(values)
    X = values_stack[:-8] #(4236, 768)
    y = keys_stack[:-8]
    
    csv_path = '../../../src/datasets/Small-Scale/PATRIC/Patric_num.csv'
    df = pd.read_csv(csv_path)
    df = df.drop_duplicates(subset='genome_id', keep='first') 
    df['genome_id'] = df['genome_id'].astype(str)
    # 使用给定的Model ID列表查找VF Cate Code
    y_output = df.set_index('genome_id').loc[y]['genome_cate_code'].values #(4230,)            
    y = y_output      
    X = np.asarray(X)
    y = np.array(y)     
    k_fold = KFold(n_splits=5, shuffle=True, random_state=16)
    predictor = SVC(probability=True)
    
    y_real = []
    y_proba = []     
    for i, (train_index, test_index) in enumerate(k_fold.split(X)):
        Xtrain, Xtest = X[train_index], X[test_index] 
        ytrain, ytest = y[train_index], y[test_index] 
        predictor.fit(Xtrain, ytrain) 
        pred_proba = predictor.predict_proba(Xtest)
        y_real.append(ytest)
        y_proba.append(pred_proba[:,1])        
        y_pred_probs = pred_proba[:,1]
        y_pred = torch.where(torch.from_numpy(y_pred_probs) > 0.5, torch.ones_like(torch.from_numpy(y_pred_probs)), torch.zeros_like(torch.from_numpy(y_pred_probs)))

    y_real = np.concatenate(y_real)
    y_proba = np.concatenate(y_proba)
    ypred = predictor.predict(Xtest)    
    report_logger = setup_logger('report_logger', './testdata_output/report_log.log')
    y_test = ytest
    accuracy = accuracy_score(y_test, y_pred)    
    print("\nTest Data Accuracy: {}".format(accuracy))
    report_logger.info("Test Data Accuracy: {}".format(accuracy))
    print("\ndata: Patric")
    report_logger.info("data: Patric")

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


def train(data_path, type, filter_num, report_logger):
    with open(data_path, 'rb') as file:
        X_train, y_train, X_val, y_val, X_test, y_test = pk.load(file)
        
    if type == "Hyena":
        nfeatures = 256
    print("successfully loaded data")
    
    train_dataset = ClassifierDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long())
    val_dataset = ClassifierDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).long())
    test_dataset = ClassifierDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).long())
    print("made datasets")
    
    # Calculate class weights from entire y (y_train + y_val + y_test)
    entire_y = np.concatenate((y_train, y_val, y_test), axis=0)
    class_count = [0] * (max([int(i) for i in entire_y]) + 1)
    y_counter = Counter(entire_y)
    for key, val in y_counter.items():
        class_count[int(key)] = val
    class_weights = 1./torch.tensor(class_count, dtype=torch.float)

    # Assign weights to train samples based on the class_weights
    train_weights = class_weights[torch.tensor([int(i) for i in y_train])].numpy()
    weighted_sampler = WeightedRandomSampler(weights=train_weights, num_samples=len(y_train), replacement=True)

    EPOCHS = 1200
    BATCH_SIZE = 5000
    LEARNING_RATE = 0.0001
    NUM_FEATURES = nfeatures
    NUM_CLASSES = len(class_count)
    train_loader = DataLoader(dataset=train_dataset,
                            batch_size=BATCH_SIZE,
                            sampler=weighted_sampler
    )
    val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE)
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
    
    # print("Begin training.")
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


    model_name = 'hyena-dna'
    figure_path = os.path.join('output_figure', model_name)
    figure_path = os.path.abspath(figure_path)
    if not os.path.exists(figure_path):
        os.makedirs(figure_path)
    print(f"\nFigure path: {figure_path}")

    plt.savefig(f"{figure_path}/acc_curve."+str(type+".png")) # train and validation accuracy curves
    plt.close(fig)
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
    print("\n Dataset: Patric")
    print("type: {}".format(type))
    print("filter_num: {}".format(filter_num))
    print("NUM_CLASSES: {}".format(NUM_CLASSES))
    report_logger.info("Dataset: Patric")
    report_logger.info("type: {}".format(type))
    report_logger.info("filter_num: {}".format(filter_num))
    report_logger.info("NUM_CLASSES: {}".format(NUM_CLASSES))

    print("\nTest Data Accuracy: {}".format(accuracy))
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
    result_path = os.path.join(report_dir, f'Patric_{type}_classification_results.csv')
    print(f"Result path: {result_path}")
    report_logger.info("Result path: %s", result_path)
    print('-'*70)
    
    results_df.to_csv(result_path, index=False)
    original_df = pd.read_csv('../../../src/datasets/Small-Scale/PATRIC/Patric_code.csv')
    encoded_df = pd.read_csv('../../../src/datasets/Small-Scale/PATRIC/Patric_num.csv')
    code_to_name = dict(zip(encoded_df['genome_cate_code'], original_df['genus']))
    results_df = pd.read_csv(result_path)
    results_df['Class Name'] = results_df['Class Name'].astype(int).map(code_to_name)
    result_path_updated = f'{figure_path}/classification_report/Patric_{type}_classification_results_updated.csv'
    results_df.to_csv(result_path_updated, index=False)
    results_df = pd.read_csv(result_path_updated)
    results_df['Correct Classification Percentage'] = (results_df['Correctly Classified'] / results_df['Total Count']) * 100
    results_df.to_csv(result_path_updated, index=False)

type = 'Hyena'

filter_num = 2
# filter_num = 3
filter_num = 5

folder_path = "./testdata_output"
report_logger = setup_logger('report_logger', './testdata_output/report_log.log')

data_path = './testdata/Hyena_small_Patric.pkl'
split_data(data_path, type, filter_num)
# train_ML(data_path)

data_path = f'./Patric_{type}.pkl'
train(data_path, type, filter_num, report_logger)