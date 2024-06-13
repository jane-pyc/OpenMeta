# usage: python pLM_embed.py <input.fasta> <output.esm.embs.pkl>
import os
GPU_NUMBER = [1]
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(s) for s in GPU_NUMBER])

import datetime
from time import *
import time
import torch
from fairscale.nn.data_parallel import FullyShardedDataParallel as FSDP
from fairscale.nn.wrap import enable_wrap, wrap
import esm
from esm import FastaBatchedDataset
import sys
from tqdm import tqdm
import os.path
import pickle as pk
import subprocess

fasta_file = sys.argv[1]
output_path = sys.argv[2]
output_dir = os.path.dirname(output_path)
subprocess.call(f"mkdir -p {output_dir}", shell=True)
toks_per_batch = 12290
dataset = FastaBatchedDataset.from_file(fasta_file)
batches = dataset.get_batch_indices(toks_per_batch, extra_toks_per_seq=1)

# init the distributed world with world_size 1
port = sys.argv[3]
url = f"tcp://localhost:{port}"
torch.distributed.init_process_group(backend="nccl", init_method=url, world_size=1, rank=0)

# download model data from the hub
model_name = "esm2_t33_650M_UR50D"
model_data, regression_data = esm.pretrained._download_model_and_regression_data(model_name)

# initialize the model with FSDP wrapper
fsdp_params = dict(
    mixed_precision=True,
    flatten_parameters=True,
    state_dict_device=torch.device("cpu"),  # reduce GPU mem usage
    cpu_offload=True,  # enable cpu offloading
)
with enable_wrap(wrapper_cls=FSDP, **fsdp_params):
    model, vocab = esm.pretrained.load_model_and_alphabet_core(
        model_name, model_data, regression_data
    )
    data_loader = torch.utils.data.DataLoader(
        dataset, collate_fn=vocab.get_batch_converter(), batch_sampler=batches
    )
    model.eval()

    # Wrap each layer in FSDP separately
    for name, child in model.named_children():
        if name == "layers":
            for layer_name, layer in child.named_children():
                wrapped_layer = wrap(layer)
                setattr(child, layer_name, wrapped_layer)
    model = wrap(model)

start = time.time()
start_time = time.time()
start_memory = torch.cuda.memory_allocated()


sequence_representations = []
with torch.no_grad():
    for batch_idx, (labels, strs, toks) in tqdm(enumerate(data_loader), total = len(data_loader)):
        toks = toks.cuda()
        toks = toks[:, :12288] #truncate 
        results = model(toks, repr_layers=[33], return_contacts=False)
        token_representations = results["representations"][33]
        for i, label in enumerate(labels):
            truncate_len = min(12288, len(strs[i]))
            sequence_representations.append((label,token_representations[i, 1 : truncate_len + 1].mean(0).detach().cpu().numpy()))

f = open(output_path, "wb")
pk.dump(sequence_representations,f)
f.close()