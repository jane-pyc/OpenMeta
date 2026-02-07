# Specifications of ARG Prediction Task

In fine-grained sequence datasets, particularly the NCRD dataset for ARG prediction tasks, we evaluate three domain-specific models: the template-matching-based RGI, the deep learning-based DeepARG.
FGBERT and RGI cover all NCRD classes, while DeepARG is limited to specific classes, likely due to limitations in its training data.

## DeepARG

Install it following the google drive link [here](https://github.com/gaarangoa/deeparg).

  
### Use conda environment
Create a virtual environment with conda:

    conda create -n deeparg_env python=2.7.18
    source activate deeparg_env

Install diamond with conda (inside virtual environment): 

    conda install -c bioconda diamond==0.9.24

Optional (used for short reads pipeline): 

    conda install -c bioconda trimmomatic
    conda install -c bioconda vsearch
    conda install -c bioconda bedtools==2.29.2
    conda install -c bioconda bowtie2==2.3.5.1
    conda install -c bioconda samtools

Install deeparg with pip and download the data required by deeparg

    pip install git+https://github.com/gaarangoa/deeparg.git
    deeparg download_data -o /path/to/local/directory/

Activate virtual environment

    conda activate deeparg_env

Deactivate the virtual environment:

    conda deactivate

```
deeparg predict --model LS -i ../../../src/datasets/Fine-Grained/NCRD-C/NCRD.fasta -o ./out2/test``$id`` -d ./deeparg3/ --type prot
```