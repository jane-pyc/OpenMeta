import os
GPU_NUMBER = [0,1,2,3,4,5,6,7]
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(s) for s in GPU_NUMBER])
import warnings
warnings.filterwarnings('ignore')
from time import *
from tqdm import trange
from Bio import SeqIO
import gensim
from gensim.models import Word2Vec
import subprocess
import torch
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, matthews_corrcoef
from BASELINES_biLSTM_att_onehot_config import Config
from BASELINES_network import BASELINES
from Utils import cal
from Utils.combine_fna import combine
from Utils.tool import data_preprocess_for_predict
import numpy as np
import pickle as pk
from tqdm import tqdm
from torch import nn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc, average_precision_score, roc_curve, classification_report, confusion_matrix
import seaborn as sns
import pandas as pd
import matplotlib as mpl
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import xgboost as xgb
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB


class BiLSTMAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(BiLSTMAttention, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.attention = nn.Linear(hidden_dim * 2, 1)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
    def forward(self, x):
        output, _ = self.lstm(x)
        attention_weights = torch.softmax(self.attention(output), dim=1)
        return output

amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
amino_acid_to_index = {aa: idx for idx, aa in enumerate(amino_acids)}
num_amino_acids = len(amino_acids)

def sequence_to_one_hot(seq, amino_acid_to_index, max_length):
    one_hot = torch.zeros(max_length, len(amino_acid_to_index))
    for i, aa in enumerate(seq[:max_length]):
        if aa in amino_acid_to_index:
            one_hot[i, amino_acid_to_index[aa]] = 1
    return one_hot

fasta_file = config.freqs_file
sequences = [str(record.seq) for record in SeqIO.parse(fasta_file, "fasta")]

max_sequence_length = 32
one_hot_sequences = torch.stack([sequence_to_one_hot(seq, amino_acid_to_index, max_sequence_length) for seq in sequences])
input_data = torch.tensor(one_hot_sequences, dtype=torch.float32)
print("input_data.shape: ",input_data.shape)
encoded_sequences_tensor = input_data 

hidden_dim = 16
output_dim = max_sequence_length  
model = BiLSTMAttention(input_dim=num_amino_acids, hidden_dim=hidden_dim, output_dim=output_dim)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

epochs = 10
for epoch in range(epochs):
    optimizer.zero_grad()
    encoded_sequences_tensor = encoded_sequences_tensor.to(device)
    outputs = model(encoded_sequences_tensor)
    outputs = outputs[:, :, :input_data.size(2)]
    loss = criterion(outputs, encoded_sequences_tensor.float()) 
    loss.backward()
    optimizer.step()

model_path = f'./model_NCyc/bilstm_att_onehot_model.pth'
torch.save(model.state_dict(), model_path)
# use trained model to predict
def predict(y_test=False):
    fasta_file = config.combined_fasta_path
    sequences = [str(record.seq) for record in SeqIO.parse(fasta_file, "fasta")]
    max_sequence_length = 32
    one_hot_sequences = torch.stack([sequence_to_one_hot(seq, amino_acid_to_index, max_sequence_length) for seq in sequences])
    input_data = torch.tensor(one_hot_sequences, dtype=torch.float32)
    print("input_data.shape: ",input_data.shape)
    encoded_sequences_tensor = input_data
    B_SIZE = 100    
    from torch.utils.data import TensorDataset, DataLoader
    dataset = TensorDataset(encoded_sequences_tensor)
    data_loader = DataLoader(dataset, batch_size=B_SIZE, shuffle=False)
    all_outputs = []
    pretrained_model = BiLSTMAttention(input_dim=num_amino_acids, hidden_dim=hidden_dim, output_dim=output_dim)
    pretrained_model.load_state_dict(torch.load(model_path))
    pretrained_model = pretrained_model.to(device)
    pretrained_model.eval()
    B_SIZE = 100
    test_pkls=[]
    test_pkls.append(str(config.freqs_file))
    all_prot_ids = []
    all_outputs = []
    with torch.no_grad():  
        for batch in data_loader:
            inputs_embeds= batch[0].type(torch.FloatTensor) 
            outputs = pretrained_model(inputs_embeds.to(device))
            all_outputs.append(outputs.cpu().detach().numpy())

    X = np.concatenate(all_outputs, axis= 0)
    X = X.mean(1)
    X = np.asarray(X)

    label_path = '../../../src/datasets/Large-Scale/NCycDB/NCyc_num.csv'
    df = pd.read_csv(label_path)
    y = df['Cate Code'].values
    y = np.array(y)
    
    k_fold = KFold(n_splits=5, shuffle=True, random_state=16) 
    predictor = RandomForestClassifier()
    for i, (train_index, test_index) in enumerate(k_fold.split(X)):
        Xtrain, Xtest = X[train_index], X[test_index] 
        ytrain, ytest = y[train_index], y[test_index] 
        predictor.fit(Xtrain, ytrain) 
        y_pred = predictor.predict(Xtest)      
        report = classification_report(ytest, y_pred, output_dict=True)
        print(f"Fold {i+1} - Overall Accuracy:", accuracy_score(ytest, y_pred))
        print(f"Fold {i+1} - Macro-Precision:", report['macro avg']['precision'])
        print(f"Fold {i+1} - Macro-Recall:", report['macro avg']['recall'])
        print(f"Fold {i+1} - Macro-F1:", report['macro avg']['f1-score'])
        print(f"Fold {i+1} - Weighted-Precision:", report['weighted avg']['precision'])
        print(f"Fold {i+1} - Weighted-Recall:", report['weighted avg']['recall'])
        print(f"Fold {i+1} - Weighted-F1:", report['weighted avg']['f1-score'])
        print('-'*50)

if __name__ == '__main__':
    config = Config()
    predict()