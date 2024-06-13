import os
GPU_NUMBER = [0,1,2,3]
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(s) for s in GPU_NUMBER])
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from collections import Counter
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import pickle as pk
from multiprocessing import Pool
from sklearn.preprocessing import label_binarize
from tqdm import tqdm  
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,precision_recall_curve,auc,average_precision_score
from sklearn.metrics import accuracy_score
import logging

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

def save_data(results_dir, type):
    f_list = os.listdir(results_dir)
    results_pkls = []    
    for pkl_f in f_list:
        if pkl_f.startswith('train.pkl.results'):
            results_pkls.append(str(os.path.join(results_dir,pkl_f)))
    first_100 = results_pkls[:10]
    embs, prot_ids= extract_ec_data_from_results_pkls_sequential(first_100, type) 
    
    pkl_filename = f"ENZYME.{type}.pkl" 
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

def split_data(data_path, type, filter_num):
    dat = pk.load(open(os.path.join(data_path, f'ENZYME.{type}.pkl'),"rb"))
    X = dat[0]
    y = dat[1] 
    y_extracted = [item.split('_')[-1] for item in y] 
    csv_path = '../../../src/datasets/Large-Scale/ENZYME/ENZYME_num.csv'
    df = pd.read_csv(csv_path)
    df = df.drop_duplicates(subset='SeqID', keep='first') 
    df['SeqID'] = df['SeqID'].astype(str)
    y_output = df.set_index('SeqID').loc[y_extracted]['Cate Code'].values 
   
    y = y_output    
    ec_counter = Counter(y)
    y_filter = [ec for ec, c in ec_counter.items() if c < filter_num ]
    y_new = np.delete(y,np.where(np.isin(y,y_filter))[0],axis =0) 
    X_new = np.delete(X,np.where(np.isin(y,y_filter))[0],axis =0)

    X_trainval, X_test, y_trainval, y_test = train_test_split(X_new, y_new, test_size=0.2, stratify=y_new, random_state=69)
    X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.1, stratify=y_trainval, random_state=21)
    
    train_split_path = f"ENZYME_train_split."+type+".pkl" 
    pkl_filepath = os.path.join(data_path, train_split_path) 
    train_dat_f = open(pkl_filepath,"wb")
    pk.dump([X_train, y_train, X_val, y_val,X_test, y_test],train_dat_f )
    train_dat_f.close()    

def train(data_path, type, filter_num, report_logger):
    X_train, y_train, X_val, y_val,X_test, y_test = pk.load(open(os.path.join(data_path, f'ENZYME_train_split.{type}.pkl'),"rb")) 
    if type == "plmglm":
        nfeatures = 2560
    else:
        nfeatures =1280
    # print("successfully loaded data")

    train_dataset = ClassifierDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long())
    val_dataset = ClassifierDataset(torch.from_numpy(X_val).float(), torch.from_numpy(y_val).long())
    test_dataset = ClassifierDataset(torch.from_numpy(X_test).float(), torch.from_numpy(y_test).long())
    entire_y = np.concatenate((y_train, y_val, y_test), axis=0)
    class_count = [0] * (max([int(i) for i in entire_y]) + 1)
    y_counter = Counter(entire_y)
    for key, val in y_counter.items():
        class_count[int(key)] = val
    class_weights = 1./torch.tensor(class_count, dtype=torch.float)

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
    print("\ntype: {}".format(type))
    print("filter_num: {}".format(filter_num))
    print("NUM_CLASSES: {}".format(NUM_FEATURES))
    print("NUM_CLASSES: {}".format(NUM_CLASSES))
    report_logger.info("type: {}".format(type))
    report_logger.info("filter_num: {}".format(filter_num))
    report_logger.info("NUM_CLASSES: {}".format(NUM_FEATURES))
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
    result_path = os.path.join(report_dir, f'ENZYME_{type}_classification_results.csv')
    print(f"Result path: {result_path}")
    report_logger.info("Result path: %s", result_path)
    print('-'*70)
    
    results_df.to_csv(result_path, index=False)
    original_df = pd.read_csv('../../../src/datasets/Large-Scale/ENZYME/ENZYME_code.csv')
    encoded_df = pd.read_csv('../../../src/datasets/Large-Scale/ENZYME/ENZYME_num.csv')
    code_to_name = dict(zip(encoded_df['Cate Code'], original_df['Cate']))
    results_df = pd.read_csv(result_path)
    results_df['Class Name'] = results_df['Class Name'].astype(int).map(code_to_name)
    result_path_updated = f'{figure_path}/classification_report/ENZYME_{type}_classification_results_updated.csv'
    results_df.to_csv(result_path_updated, index=False)
    results_df = pd.read_csv(result_path_updated)
    results_df['Correct Classification Percentage'] = (results_df['Correctly Classified'] / results_df['Total Count']) * 100
    results_df.to_csv(result_path_updated, index=False)

type = 'plmglm'
# type = 'plm'

filter_num = 2
filter_num = 3
# filter_num = 5

def recursive_search(folder_path, type, report_logger):
    for root, dirs, files in os.walk(folder_path):
        for dir in dirs:
            if dir == "results":
                results_folder = os.path.join(root, dir)                
                save_data(results_folder, type)                
                split_data(results_folder, type, filter_num)    
                train(results_folder, type, filter_num, report_logger)                      
folder_path = "./output/"
report_logger = setup_logger('report_logger', './output_Triplet2/report_log2.log')
recursive_search(folder_path, type, report_logger)
if type != "plm" and type != "plmglm":
    print("You must specify data type: glm or plmglm")