import pandas as pd
import numpy as np
import joblib
from utility import extract
from esm.pretrained import load_model_and_alphabet_local
import pdb

def predict(in_fasta, batch_size=10, maxlen = 200, min_prob = 0.5, arg_model='models/arg_model.pkl',
            cat_model='models/cat_model.pkl',cat_index='models/Category_Index.csv',output_file='higarg_out.tsv'):
    ## 1. load arg model and category model and category index
    arg_model = joblib.load(arg_model)
    cat_model = joblib.load(cat_model)
    cat_index = np.loadtxt(cat_index,dtype = str,delimiter = ",").tolist()

    # 2. generating the embedding representation
    print("Loading the ESM-1b model for protein embedding ...")
    try:
        path = '/root/.cache/torch/hub/checkpoints/esm1b_t33_650M_UR50S.pt'
        model, alphabet = load_model_and_alphabet_local(path)
        model.eval()
    except IOError:
        print("The ESM-1b model is not accessible.")
    
    seq_id, embedding_res = extract(in_fasta, alphabet, model, repr_layers = [32], 
                                    batch_size = batch_size, max_len= maxlen)
    seq_num = len(seq_id)
    cat_num = len(cat_index)
    pred_res = pd.DataFrame({'seq_id':seq_id, 'pred':''})
    pred_res = pd.concat([pred_res, pd.DataFrame(data = np.zeros((seq_num,cat_num+1),dtype='float64'),
                     columns= ['ARG']+cat_index)], axis = 1)
    pred_res['ARG'] = arg_model.predict_proba(embedding_res)[:,1]
    arg_ind = np.where(pred_res['ARG']>min_prob)[0].tolist()
    if len(arg_ind) > 0:
        cat_out = cat_model.predict_proba(embedding_res[arg_ind,])
    for i in range(len(cat_out)):
        pred_res.iloc[arg_ind, i + 3] = cat_out[i][:, 1]
    for i in arg_ind:
        cats = [cat_index[k] for k, v in enumerate(pred_res.iloc[i, 3:]) if v >= 0.5]
        pred_res.iloc[i, 1] = ';'.join(cats)
    pred_res.to_csv(output_file, sep = '\t', index=0)
