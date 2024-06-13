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
from sklearn.metrics import accuracy_score
from BASELINES_biLSTM_att_w2v_config import Config
from BASELINES_network import BASELINES
from Utils import cal
from Utils.combine_fna import combine
import numpy as np
import pickle as pk
from tqdm import tqdm
from torch import nn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
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
        self.embedding = nn.Embedding(len(vocab) + 1, input_dim)
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.attention = nn.Linear(hidden_dim * 2, 1)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
    
    def forward(self, x):
        embedded = self.embedding(x)
        output, _ = self.lstm(embedded)
        attention_weights = torch.softmax(self.attention(output).squeeze(-1), dim=1)
        context_vector = output * attention_weights.unsqueeze(-1)
        output = self.fc(context_vector)
        return output

fasta_file = config.freqs_file
sequences = [str(record.seq) for record in SeqIO.parse(fasta_file, "fasta")]
embedding_dim = 100 
word2vec_model = Word2Vec(sentences=[list(seq) for seq in sequences], vector_size=embedding_dim, window=5, min_count=1, sg=0)

vocab = word2vec_model.wv.key_to_index
vocab_size = len(vocab)
encoded_sequences = []
for seq in sequences:
    encoded_seq = [vocab[aa] for aa in seq if aa in vocab]
    encoded_sequences.append(encoded_seq)
sequence_lengths = [len(seq) for seq in encoded_sequences]
max_sequence_length = 32
padded_sequences = []
for seq in encoded_sequences:
    if len(seq) > max_sequence_length:
        padded_seq = seq[:max_sequence_length]
    else:
        pad_length = max_sequence_length - len(seq)
        padded_seq = seq + [0] * pad_length 
    padded_sequences.append(padded_seq)
encoded_sequences_tensor = torch.tensor(padded_sequences, dtype=torch.long)

input_dim = embedding_dim
hidden_dim = 16
output_dim = max_sequence_length  
model = BiLSTMAttention(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
epochs = 10
for epoch in range(epochs):
    optimizer.zero_grad()
    encoded_sequences_tensor = encoded_sequences_tensor.to(device)
    outputs = model(encoded_sequences_tensor)
    loss = criterion(outputs, encoded_sequences_tensor.long()) 
    loss.backward()
    optimizer.step()
model_path = f'./model_NCyc/pretrained_bilstm_att_w2v_input_size_{embedding_dim}_output_size_{max_sequence_length}_model.pth'
torch.save(model.state_dict(), model_path)
# use trained model to predict
def predict(y_test=False):
    combine(config.raw_fasta_path, config.combined_fasta_path)
    names = cal.biLSTM_att_w2v(config.combined_fasta_path, config.num_procs, config.freqs_file, max_sequence_length)
    pretrained_model = BiLSTMAttention(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)
    pretrained_model.load_state_dict(torch.load(model_path))
    pretrained_model = pretrained_model.to(device)
    pretrained_model.eval()

    B_SIZE = 100
    test_pkls=[]
    test_pkls.append(str(config.freqs_file))
    all_outputs = []
        
    """examines the model"""
    
    for pkl_f in test_pkls:
        pickle_file =  open(pkl_f, 'rb')
        dataset = pk.load(pickle_file)
        if B_SIZE < len(dataset):
            loader = torch.utils.data.DataLoader(dataset, batch_size =B_SIZE, shuffle=False, drop_last=False)
        else:
            loader = torch.utils.data.DataLoader(dataset, batch_size =B_SIZE, shuffle=False)
        loop = tqdm(loader, leave=True)
        for batch in loop:
            inputs_embeds= batch[1].type(torch.FloatTensor) 
            inputs_embeds = inputs_embeds.long() 
            with torch.no_grad(): 
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