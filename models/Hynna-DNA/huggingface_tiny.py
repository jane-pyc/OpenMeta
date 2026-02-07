import json
import os
import subprocess
import torch
# import transformers
from transformers import PreTrainedModel
import re
from standalone_hyenadna import HyenaDNAModel
from standalone_hyenadna import CharacterTokenizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    print("CUDA is not available. Using CPU.")
    device = torch.device("cpu")

def read_fasta_file(file_path):
    sequences = {}
    current_seq_id = None
    current_seq = []

    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('>'):
                if current_seq_id is not None:
                    sequences[current_seq_id] = ''.join(current_seq)
                current_seq_id = line[1:].strip()
                current_seq = []
            else:
                current_seq.append(line.strip())
    if current_seq_id is not None:
        sequences[current_seq_id] = ''.join(current_seq)
    return sequences
# helper 1
def inject_substring(orig_str):
    """Hack to handle matching keys between models trained with and without
    gradient checkpointing."""

    # modify for mixer keys
    pattern = r"\.mixer"
    injection = ".mixer.layer"

    modified_string = re.sub(pattern, injection, orig_str)

    # modify for mlp keys
    pattern = r"\.mlp"
    injection = ".mlp.layer"

    modified_string = re.sub(pattern, injection, modified_string)

    return modified_string

# helper 2
def load_weights(scratch_dict, pretrained_dict, checkpointing=False):
    """Loads pretrained (backbone only) weights into the scratch state dict."""

    # loop thru state dict of scratch
    # find the corresponding weights in the loaded model, and set it

    # need to do some state dict "surgery"
    for key, value in scratch_dict.items():
        if 'backbone' in key:
            # the state dicts differ by one prefix, '.model', so we add that
            key_loaded = 'model.' + key
            # breakpoint()
            # need to add an extra ".layer" in key
            if checkpointing:
                key_loaded = inject_substring(key_loaded)
            try:
                scratch_dict[key] = pretrained_dict[key_loaded]
            except:
                raise Exception('key mismatch in the state dicts!')

    # scratch_dict has been updated
    return scratch_dict

class HyenaDNAPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    base_model_prefix = "hyenadna"

    def __init__(self, config):
        pass

    def forward(self, input_ids, **kwargs):
        return self.model(input_ids, **kwargs)

    @classmethod
    def from_pretrained(cls,
                        path,
                        model_name,
                        download=False,
                        config=None,
                        device='cpu',
                        use_head=False,
                        n_classes=2,
                      ):
        # first check if it is a local path
        # download = False
        pretrained_model_name_or_path = os.path.join(path, model_name)
        if os.path.isdir(pretrained_model_name_or_path) and download == False:
            if config is None:
                config = json.load(open(os.path.join(pretrained_model_name_or_path, 'config.json')))
        else:
            hf_url = f'https://huggingface.co/LongSafari/{model_name}'

            subprocess.run(f'rm -rf {pretrained_model_name_or_path}', shell=True)
            command = f'mkdir -p {path} && cd {path} && git lfs install && git clone {hf_url}'
            subprocess.run(command, shell=True)

            if config is None:
                config = json.load(open(os.path.join(pretrained_model_name_or_path, 'config.json')))

        scratch_model = HyenaDNAModel(**config, use_head=use_head, n_classes=n_classes)  # the new model format
        loaded_ckpt = torch.load(
            os.path.join(pretrained_model_name_or_path, 'weights.ckpt'),
            map_location=torch.device(device)
        )

        # need to load weights slightly different if using gradient checkpointing
        if config.get("checkpoint_mixer", False):
            checkpointing = config["checkpoint_mixer"] == True or config["checkpoint_mixer"] == True
        else:
            checkpointing = False

        # grab state dict from both and load weights
        state_dict = load_weights(scratch_model.state_dict(), loaded_ckpt['state_dict'], checkpointing=checkpointing)

        # scratch model has now been updated
        scratch_model.load_state_dict(state_dict)
        # print("Loaded pretrained weights ok!")
        return scratch_model


import json
import os
import subprocess
# import transformers
from transformers import PreTrainedModel
import logging
import time
import pickle as pkl
import psutil
# os.environ["http_proxy"] = "http://www.zangzelin.fun:4081"
# os.environ["https_proxy"] = "http://www.zangzelin.fun:4081"
# os.environ["all_proxy"] = "socks5://www.zangzelin.fun:4082"
# os.environ["CUDA_VISIBLE_DEVICES"] = "5"

# subprocess.run(["python", "huggingface_tiny.py"])

# del os.environ["http_proxy"]
# del os.environ["https_proxy"]
# del os.environ["all_proxy"]


logging.basicConfig(filename='model_performance.log', level=logging.INFO)
logger = logging.getLogger(__name__)


def inference_single(sequence):

    '''
    this selects which backbone to use, and grabs weights/ config from HF
    4 options:
      'hyenadna-tiny-1k-seqlen'   # fine-tune on colab ok
      'hyenadna-small-32k-seqlen'
      'hyenadna-medium-160k-seqlen'  # inference only on colab
      'hyenadna-medium-450k-seqlen'  # inference only on colab
      'hyenadna-large-1m-seqlen'  # inference only on colab
    '''

    # you only need to select which model to use here, we'll do the rest!
    pretrained_model_name = 'hyenadna-small-32k-seqlen'
    model_name = pretrained_model_name

    max_lengths = {
        'hyenadna-tiny-1k-seqlen': 1024,
        'hyenadna-small-32k-seqlen': 32768,
        'hyenadna-medium-160k-seqlen': 160000,
        'hyenadna-medium-450k-seqlen': 450000,  # T4 up to here
        'hyenadna-large-1m-seqlen': 1_000_000,  # only A100 (paid tier)
    }

    max_length = max_lengths[pretrained_model_name]  # auto selects

    # data settings:
    use_padding = True
    rc_aug = False  # reverse complement augmentation
    add_eos = False  # add end of sentence token

    # we need these for the decoder head, if using
    use_head = False
    n_classes = 2  # not used for embeddings only

    # you can override with your own backbone config here if you want,
    # otherwise we'll load the HF one in None
    backbone_cfg = None

    device = 'cuda' if torch.cuda.is_available() else 'cpu'   

    # instantiate the model (pretrained here)
    if pretrained_model_name in ['hyenadna-tiny-1k-seqlen',
                                 'hyenadna-small-32k-seqlen',
                                 'hyenadna-medium-160k-seqlen',
                                 'hyenadna-medium-450k-seqlen',
                                 'hyenadna-large-1m-seqlen']:
        # use the pretrained Huggingface wrapper instead
        model = HyenaDNAPreTrainedModel.from_pretrained(
            './checkpoints',
            pretrained_model_name,
            download=False,
            config=backbone_cfg,
            device=device,
            use_head=use_head,
            n_classes=n_classes,
        )
    
    # from scratch
    elif pretrained_model_name is None:
        model = HyenaDNAModel(**backbone_cfg, use_head=use_head, n_classes=n_classes)
    
    # prep model and forward
    model.to(device)
    model.eval()

    # create tokenizer
    tokenizer = CharacterTokenizer(
        characters=['A', 'C', 'G', 'T', 'N'],  # add DNA characters, N is uncertain
        model_max_length=max_length + 2,  # to account for special tokens, like EOS
        add_special_tokens=False,  # we handle special tokens elsewhere
        padding_side='left', # since HyenaDNA is causal, we pad on the left
    )

    #### Single embedding example ####

    # create a sample 450k long, prepare  
    # tok_seq = tokenizer(sequence)
    tok_seq = tokenizer(sequence, padding="max_length", truncation=True, max_length=max_length)
    tok_seq = tok_seq["input_ids"]  # grab ids

    # place on device, convert to tensor
    tok_seq = torch.LongTensor(tok_seq).unsqueeze(0)  # unsqueeze for batch dim
    tok_seq = tok_seq.to(device)

    # prep model and forward
    model.to(device)
    model.eval()
    with torch.inference_mode():
        embeddings = model(tok_seq)
    
    return embeddings[0].mean(0)

# CARD
fasta_file = '../../../src/datasets/Small-Scale/CARD-A/CARD.fasta'
output_path = './testdata/Hyena_small_CARD.pkl'

# Operons
fasta_file = '../../../src/datasets/Small-Scale/E-K12/E-K12.fasta'
output_path = './testdata/Hyena_small_Operons.pkl'
sequences = read_fasta_file(fasta_file)
print(f"Total sequences processed: {len(sequences)}")


# VFDB
fasta_file = '../../../src/datasets/Large-Scale/VFDB/VFDB.fasta'
output_path = './testdata/Hyena_small_VFDB.pkl'
sequences = read_fasta_file(fasta_file)
print(f"Total sequences processed: {len(sequences)}")


# # ENZYME
fasta_file = '../../../src/datasets/Large-Scale/ENZYME/ENZYME.fasta'
output_path = './testdata/Hyena_small_ENZYME.pkl'
sequences = read_fasta_file(fasta_file)
print(f"Total sequences processed: {len(sequences)}")


# # Patric
fasta_file = '../../../src/datasets/Small-Scale/PATRIC/Patric.fasta'
output_path = './testdata/Hyena_small_Patric.pkl'
sequences = read_fasta_file(fasta_file)
print(f"Total sequences processed: {len(sequences)}")


# # N-Cycle
fasta_file = '../../../src/datasets/Large-Scale/NCycDB/NCyc.fasta'
output_path = './testdata/Hyena_small_NCyc.pkl'
sequences = read_fasta_file(fasta_file)
print(f"Total sequences processed: {len(sequences)}")

