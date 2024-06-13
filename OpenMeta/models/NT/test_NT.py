import haiku as hk
import jax
import jax.numpy as jnp
import psutil
from nucleotide_transformer.pretrained import get_pretrained_model
import pickle as pkl
import os
import time
import logging
import torch
import subprocess

print("JAX backend:", jax.lib.xla_bridge.get_backend().platform)
GPU_NUMBER = [0,1,2,3,4,5,6,7]
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3,4,5,6,7"  

logging.basicConfig(filename='model_performance.log', level=logging.INFO)
logger = logging.getLogger(__name__)

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]=".XX"
os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"]="platform"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print("CUDA is available. Using GPU.")
    device = torch.device("cuda")
else:
    print("CUDA is not available. Using CPU.")
    device = torch.device("cpu")

def parse_fasta_file(file_path):
    sequences = []
    with open(file_path, 'r') as file:
        current_seq = ''
        for line in file:
            if line.startswith('>'):
                # if current_seq:
                if current_seq and all(c in 'AGCT' for c in current_seq):
                    sequences.append(current_seq)
                    current_seq = ''
            else:
                current_seq += line.strip()
        if current_seq and all(c in 'AGCT' for c in current_seq):
            sequences.append(current_seq)
    return sequences

# Operons
fasta_file = '../../../src/datasets/Small-Scale/E-K12/E-K12.fasta'
output_path = './testdata/NT_Operons1.pkl'

# CARD
fasta_file = '../../../src/datasets/Small-Scale/CARD-A/CARD.fasta'
output_path = './testdata/NT_CARD.pkl'

# VFDB
fasta_file = '../../../src/datasets/Large-Scale/VFDB/VFDB.fasta'
output_path = './testdata/NT_VFDB2.pkl'

# # ENZYME
fasta_file = '../../../src/datasets/Large-Scale/ENZYME/ENZYME.fasta'
output_path = './testdata/NT_ENZYME.pkl'


# # Patric
fasta_file = '../../../src/datasets/Small-Scale/PATRIC/Patric.fasta'
output_path = './testdata/NT_Patric1.pkl'


# N-Cycle
fasta_file = '../../../src/datasets/Large-Scale/NCycDB/NCyc.fasta'
output_path = './testdata/NT_NCyc1.pkl'

total_processing_time = 0
start_time = time.time()
sequences = parse_fasta_file(fasta_file)
print(f"Total sequences processed: {len(sequences)}")

model_names = [
    "50M_multi_species_v2"
]
max_length = 32
model_name = "50M_multi_species_v2"
fasta_file_path = fasta_file
parameters, forward_fn, tokenizer, config = get_pretrained_model(
    model_name=model_name,
    embeddings_layers_to_save=(20,),
    max_positions=32,
)
forward_fn = hk.transform(forward_fn)
tokens_ids = []
for sequence in sequences:    
    truncated_sequence = sequence[:max_length]
    tokens_ids.extend([b[1] for b in tokenizer.batch_tokenize([truncated_sequence])])
tokens = jnp.asarray(tokens_ids, dtype=jnp.int32)
# Initialize random key
random_key = jax.random.PRNGKey(0)
# Infer
outs = forward_fn.apply(parameters, random_key, tokens)
# print(f"Model: {model_name}, Embeddings Shape: {outs['embeddings_20'].shape}")
outs = outs['logits'].mean(1) #[4743,4107]
end_time = time.time()
f = open(output_path,'wb')
pkl.dump(outs, f)
f.close()