## ViBE: a hierarchical BERT model to identify viruses using metagenome sequencing data

This repository includes the implementation of 'ViBE: a hierarchical BERT model to identify viruses using metagenome sequencing data'. Hints can be obtained from the official [website](https://github.com/DMnBI/ViBE).

## Environment Setup
We strongly recommend you to use python virtual environment with [Anaconda](https://www.anaconda.com)/[Miniconda](https://docs.conda.io/en/latest/miniconda.html). Moreover, this model works in practical time on GPU/TPU machine. PLEASE use one or more NVIDIA GPUs (It works without errors using just the CPU, but it will take an astronomical amount of time). The details of the machine/environment we used are as follows:

* NVIDIA A100 with 40GB graphic memory
* CUDA 11.0
* python 3.6.13
* pytorch 1.10.0
* [transformers 4.11.3 - huggingface](https://huggingface.co/docs/transformers/index)
* [datasets 1.15.1 - huggingface](https://huggingface.co/docs/datasets/)


```
conda update conda (optional)
conda create -n vibe python=3.6
conda activate vibe
conda install -c huggingface transformers datasets
conda install -c pytorch pytorch torchvision cudatoolkit
conda install scikit-learn
```

## Install ViBE and download models

The source files and useful scripts are in this repository. The pre-trained and fine-tuned models have been uploaded on **Google Drive** since the size of each model is larger than 100MB. PLEASE make sure to download models after cloning this repository.

```
git clone https://github.com/DMnBI/ViBE.git
cd ViBE
chmod +x src/vibe
```

The `vibe` script in the `src` directory is an executable python script. No additional installation is required.

**Download Models**

Please download the model you need through the link below and save them in the `models` directory. You can also download models using the `download_models.py` script in the `scripts` directory. 

```
chmod u+x scripts/gdown.sh
python scripts/download_models.py -d all -o ./models
```

**Pre-trained model**

* [pre-trained](https://drive.google.com/file/d/100EITt7ZmyjkBl_X1kJ83nfV5jpK_ED1/view?usp=sharing)

**Domain-level classifier**

* [BPDR.150bp](https://drive.google.com/file/d/1nSTwkvfeJ5VTs2__FOIVW9IO-L8iQZid/view?usp=sharing)
* [BPDR.250bp](https://drive.google.com/file/d/1WdawuAiz1E4CYwrtjvd24dNFHUjns9ZZ/view?usp=sharing)

**Order-level classifiers**

* [DNA.150bp](https://drive.google.com/file/d/1HrFwr-VQrUHA9vdUowQtOgTCxb6IBA9u/view?usp=sharing)
* [DNA.250bp](https://drive.google.com/file/d/1C-MMl-tMuTJnEkzTrt7EEIRJKB5OqZha/view?usp=sharing)
* [RNA.150bp](https://drive.google.com/file/d/1JHD146DDftVLmM8yecNxjxR28v8SUtGt/view?usp=sharing)
* [RNA.250bp](https://drive.google.com/file/d/1c_jKpqDE8L7hZOKkiTPai53FNzYVGscp/view?usp=sharing)

## Data Preparation
Convert your FASTA files into a CSV format that ViBE can process:
```
python scripts/seq2kmer_doc.py -i your_data.fasta -o your_output.csv -k 4
```

## Model Fine-Tuning
To fine-tune the model with your prepared data, execute the finetune.sh script located in the root directory of the project:
```
./finetune.sh
```

## Running Predictions
After fine-tuning, you can use the trained model to make predictions on new data:
```
src/vibe predict --model models/fine-tuned/model.pth --sample_file your_output.csv --output_dir results
```