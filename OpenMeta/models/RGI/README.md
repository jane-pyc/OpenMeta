# Specifications of ARG Prediction Task

In fine-grained sequence datasets, particularly the NCRD dataset for ARG prediction tasks, we evaluate three domain-specific models: the template-matching-based RGI, the deep learning-based DeepARG.
FGBERT and RGI cover all NCRD classes, while DeepARG is limited to specific classes, likely due to limitations in its training data.

## RGI

Install it following the google drive link [here](https://github.com/arpcard/rgi).

```
mamba search --channel conda-forge --channel bioconda --channel defaults rgi

mamba create --name rgi --channel conda-forge --channel bioconda --channel defaults rgi

mamba install --channel conda-forge --channel bioconda --channel defaults rgi

mamba install --channel conda-forge --channel bioconda --channel defaults rgi=5.1.1

mamba remove rgi

rgi main --input_sequence ../../../src/datasets/Fine-Grained/NCRD-C/NCRD.fasta --output_file ./test``$id``  -n 8 --clean -t protein
```