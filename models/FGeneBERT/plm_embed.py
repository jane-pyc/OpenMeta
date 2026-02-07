import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--fasta', type=str, required=True, help='Input fasta file')
parser.add_argument('--out', type=str, required=True, help='Output pkl file')
parser.add_argument('--port', type=int, default=23442, help='Port for distributed training')
parser.add_argument('--gpu', type=str, default='1', help='GPU device ids')
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

import datetime
from time import *
import time
import torch
from fairscale.nn.data_parallel import FullyShardedDataParallel as FSDP
from fairscale.nn.wrap import enable_wrap, wrap
import esm
from esm import FastaBatchedDataset
from tqdm import tqdm
import pickle as pk
import subprocess

output_dir = os.path.dirname(args.out)
if output_dir:
    subprocess.call(f"mkdir -p {output_dir}", shell=True)

toks_per_batch = 12290
dataset = FastaBatchedDataset.from_file(args.fasta)
batches = dataset.get_batch_indices(toks_per_batch, extra_toks_per_seq=1)

url = f"tcp://localhost:{args.port}"
torch.distributed.init_process_group(backend="nccl", init_method=url, world_size=1, rank=0)

model_name = "esm2_t33_650M_UR50D"
model_data, regression_data = esm.pretrained._download_model_and_regression_data(model_name)

fsdp_params = dict(
    mixed_precision=True,
    flatten_parameters=True,
    state_dict_device=torch.device("cpu"),
    cpu_offload=True,
)
with enable_wrap(wrapper_cls=FSDP, **fsdp_params):
    model, vocab = esm.pretrained.load_model_and_alphabet_core(
        model_name, model_data, regression_data
    )
    data_loader = torch.utils.data.DataLoader(
        dataset, collate_fn=vocab.get_batch_converter(), batch_sampler=batches
    )
    model.eval()

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
    for batch_idx, (labels, strs, toks) in tqdm(enumerate(data_loader), total=len(data_loader)):
        toks = toks.cuda()
        toks = toks[:, :12288]
        results = model(toks, repr_layers=[33], return_contacts=False)
        token_representations = results["representations"][33]
        for i, label in enumerate(labels):
            truncate_len = min(12288, len(strs[i]))
            sequence_representations.append((label, token_representations[i, 1:truncate_len + 1].mean(0).detach().cpu().numpy()))

with open(args.out, "wb") as f:
    pk.dump(sequence_representations, f)
