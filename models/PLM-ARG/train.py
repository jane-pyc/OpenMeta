import torch
from esm.pretrained import load_model_and_alphabet_local
import joblib
import numpy as np
from xgboost import XGBClassifier
from utility import extract, get_label
from sklearn.multioutput import MultiOutputClassifier

def train(in_fasta, maxlen = 200, min_seq = 50, batch_size = 10,
          arg_model = 'models/arg_model.pkl', cat_model = 'models/cat_model.pkl',
          cat_index = 'models/Category_Index.csv'):
    print("Loading the ESM-1b model for protein embedding ...")
    try:
        model, alphabet = load_model_and_alphabet_local('models/esm1b_t33_650M_UR50S.pt')
        model.eval()
    except IOError:
        print("The ESM-1b model is not accessible.")

    # extract the embedding vectors
    print("Generating embedding representation for each protein sequence ...")
    seq_id, embedding_res = extract(in_fasta, alphabet, model, repr_layers = [32], 
                                    batch_size = batch_size, max_len= maxlen)
    # get categories for training
    print("Get the resistance categories with more than "+ str(min_seq) + " proteins ...")
    Label_ID, ARG_Category = get_label(seq_id)
    np.savetxt(cat_index, ARG_Category, delimiter=",", fmt='%s')
    # training with XGBoost
    X = embedding_res
    Y = Label_ID
    ## 1. train model for ARG identification
    print("Training for ARG identification ...")
    model1 = XGBClassifier(learning_rate=0.1, objective='binary:logistic',
                           max_depth = 7, n_estimators = 200)
    model1.fit(X, Y[:,0])
    joblib.dump(model1, arg_model)
    print("Training for resistance category classification ...")
    arg_ind = Y[:,0] == 1
    ARG_X = X[arg_ind,:]
    ARG_Y = Y[arg_ind,1:]
    model2 = MultiOutputClassifier(XGBClassifier(learning_rate=0.1, 
                                                    objective='binary:logistic',
                                                    max_depth = 7, n_estimators = 200))
    model2.fit(ARG_X, ARG_Y)
    joblib.dump(model2, cat_model)
    
   