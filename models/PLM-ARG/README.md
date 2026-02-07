# Specifications of ARG Prediction Task

In fine-grained sequence datasets, particularly the NCRD dataset for ARG prediction tasks, we evaluate three domain-specific models: the template-matching-based RGI, the deep learning-based DeepARG.
FGBERT and RGI cover all NCRD classes, while DeepARG is limited to specific classes, likely due to limitations in its training data.

## PLM-ARG

To use the PLM-ARG codes, you first need to download the 'esm1b_t33_650M_UR50S.pt' and put it under the fold of 'models/'.
Hints can be obtained from the official PLM-ARG [website](https://github.com/Junwu302/PLM-ARG).

```
python plm_arg.py predict -i ../../../src/datasets/Fine-Grained/NCRD-C/NCRD.fasta -o plm_arg_res.tsv --arg-model ./models/arg_model.pkl  --cat-model ./models/cat_model.pkl --cat-index ./models/Category_Index.csv
```