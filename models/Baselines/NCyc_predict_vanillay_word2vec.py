import os
GPU_NUMBER = [0,1,2,3,4,5,6,7]
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(s) for s in GPU_NUMBER])
import warnings
warnings.filterwarnings('ignore')
from time import *
from Bio import SeqIO
import torch
from sklearn.metrics import accuracy_score
from BASELINES_LSTM_w2v_config import Config
from BASELINES_network import BASELINES
from Utils import cal
from Utils.combine_fna import combine
from Utils.tool import data_preprocess_for_predict
import numpy as np
import pickle as pk
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import xgboost as xgb

class TransformerAutoencoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, nhead, num_encoder_layers, num_decoder_layers):
        super(TransformerAutoencoder, self).__init__()
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=input_dim, nhead=nhead, dim_feedforward=hidden_dim),
            num_layers=num_encoder_layers)
        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=input_dim, nhead=nhead, dim_feedforward=hidden_dim),
            num_layers=num_decoder_layers)
        self.fc_out = nn.Linear(input_dim, input_dim) 

    def forward(self, src):
        src = src.permute(1, 0, 2) 
        memory = self.encoder(src)
        output = self.decoder(src, memory)
        output = self.fc_out(output)
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

embedding_dim = 20
input_dim = embedding_dim
hidden_dim = 16 
output_dim = max_sequence_length  
nhead = 2
num_encoder_layers = 4
num_decoder_layers = 4
model = TransformerAutoencoder(input_dim=embedding_dim, hidden_dim=hidden_dim, nhead=nhead, num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = model.to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
epochs=100
for epoch in range(epochs):
    optimizer.zero_grad()
    outputs = model(encoded_sequences_tensor.to(device)).permute(1,0,2)
    loss = criterion(outputs, encoded_sequences_tensor.to(device))
    loss.backward()
    optimizer.step()

model_path = f'./model_NCyc/vanilla_Transformer_onehot_input_size_{embedding_dim}_output_size_{max_sequence_length}_model.pth'
torch.save(model.state_dict(), model_path)

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
    
    pretrained_model = TransformerAutoencoder(input_dim=embedding_dim, hidden_dim=hidden_dim, nhead=nhead, num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers)
    pretrained_model.load_state_dict(torch.load(model_path))
    pretrained_model = pretrained_model.to(device)
    pretrained_model.eval()
    
    """examines the model"""    
    with torch.no_grad():  
        for batch in data_loader:
            inputs_embeds= batch[0].type(torch.FloatTensor) 
            outputs = pretrained_model(inputs_embeds.to(device)).permute(1,0,2)
            all_outputs.append(outputs.cpu().detach().numpy())

    X = np.concatenate(all_outputs, axis= 0)
    X = X.mean(1)
    X = np.asarray(X)
    label_path = '../../../src/datasets/Large-Scale/NCycDB/NCyc_num.csv'
    df = pd.read_csv(label_path)
    y = df['Cate Code'].values
    y = np.array(y)     
    k_fold = KFold(n_splits=5, shuffle=True, random_state=16) 
    predictor = xgb.XGBClassifier()
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