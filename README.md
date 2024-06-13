# OpenMeta: A Comprehensive Multi-Task Benchmark for Metagenomics Understanding

This repository contains the code for OpenMeta, a benchmark for metagenomics understanding algorithms. Training is implemented with the [PyTorch Lightning](https://www.pytorchlightning.ai/) framework, logging with [Weights and Biases](https://wandb.ai/).

This is **OpenMeta**:
<p align="center">
  <img src="./src/figure/figure1.png" /> 
</p>


## Usage
We provide support for Python >= 3.10. Install requirements with
```bash
python -m pip install -r requirements.txt
```
For instructions on how to install PyTorch versions compatible with your CUDA versions, see [pytorch.org](https://pytorch.org/). We support Torch 2.1.2.


## Datasets
We cover three types of sequence datasets:
- Small-Scale Classification
    - E-K12
    - CARD-AMR Gene Families (CARD-A)
    - CARD-Drug Classes (CARD-D)
    - CARD-Resistance Mechanisms (CARD-R)
    - PATRIC core dataset
- Large-Scale Classification
    - ENZYME
    - VFDB
    - NCycDB
- Fine-Grained Classification
    - NCRD-Gene Families (NCRD-N)
    - NCRD-Gene Name (NCRD-F)
    - NCRD-Resistance (NCRD-C)
    - NCRD-Mechanisms (NCRD-R)

and hierarchiccal datasets:
- Hierarchical Classification
    - Cirrhosis
    - T2D


| Task Type | Dataset | Task | Task Category | #Genome | #Category |
| --- | --- | --- | --- | --- | --- |
| Small-Scale | E-K12 | Gene Structure Analysis | Gene-pair Cls. | 4,315 | 1,379 |
|  | CARD-A AMR Gene Family | Functional Gene Prediction | Gene-wise Cls. | 1,966 | 269 |
|  | CARD-D Drug Class | Functional Gene Prediction |  | 1,966 | 37 |
|  | CARD-R Resistance Mechanism | Functional Gene Prediction |  | 1,966 | 7 |
|  | PATRIC Pathogenic Genes | Pathogenicity Potential Assessment |  | 5,000 | 110 |
| Large-Scale | ENZYME | Functional Gene Prediction | Gene-wise Cls. | 5,761 | 7 |
|  | VFDB | Functional Gene Prediction |  | 8,945 | 15 |
|  | NCycDB Nitrogen Cycling Genes | Nitrogen Cycle Prediction |  | 213,501 | 68 |
| Fine-Grained | NCRD-N Gene Name | Functional Gene Prediction | Gene-wise Cls. | 104,363 | 1912 |
|  | NCRD-F Gene Family | Functional Gene Prediction |  | 104,363 | 420 |
|  | NCRD-C Categories | Functional Gene Prediction |  | 104,363 | 29 |
|  | NCRD-R Resistance Mechanism | Functional Gene Prediction |  | 104,363 | 10 |


| Type | Dataset | Category | #Seq. | Taxonomic Levels |  |  |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  |  |  |  | Kindom | Phylum | Class | Order | Family | Genus | Specialized |
| Hierarchical Classification | Cirrhosis | Metagenome-wise Classification | 542 | 3 | 15 | 27 | 40 | 76 | 186 | 531 |
|  | T2D |  | 606 | 3 | 17 | 29 | 48 | 94 | 216 | 587 |


We provide a set of datasets (in `src/datasets`) including small-scale, large-scale and fine-grained sequence data, as well as structural data, covering the fields of gene structure analysis, functional gene prediction, pathogenicity assessment, nitrogen cycle prediction, and disease phenotype prediction.

Each folder contains the original .fasta file of a dataset, and the corresponding annotation file .annot or .csv. The *_num.csv and *_code.csv files have two columns. The first column is the sequence ID, and the second column is the corresponding annotation ID or annotation name.
Examples:
| .fasta | *_code.csv | *_num.csv |
| --- | --- | --- |
| >Sequence_0_469009.4 MAGKQITKNPTTKPYKVGTTHPKNGDVVIFENPNYNNPNVSEQERQAAQKYRFIIHRVVYNDSKNELIGTWGDANDTQHKYEQKIPYHKIVGKYVCKDRYIFPSIKNFFSDLLILVLSLLFYLKIPYICF | genome_id,genus 469009.4,Candidatus Phytoplasma | genome_id,genome_cate_code 469009.4,0 |


## Overview of Models:

| Model | Pre-Trained Model | Not Pre-Trained Model | General Model | Specialized Model | Sequence-based Model | Structure-based Model | Year | Link | Description |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SVM |  | &#10003; | &#10003; |  | &#10003; |  |  |  | Machine Learning |
| RF |  | &#10003; | &#10003; |  | &#10003; |  |  |  | Machine Learning |
| AdaBoost |  | &#10003; | &#10003; |  | &#10003; |  |  |  | Machine Learning |
| CNN |  | &#10003; | &#10003; |  | &#10003; |  |  |  | Deep Learning |
| LSTM |  | &#10003; | &#10003; |  | &#10003; |  |  |  | Deep Learning |
| Vanilla Transformer |  | &#10003; | &#10003; |  | &#10003; |  |  |  | Deep Learning |
| FGBERT | &#10003; |  | &#10003; |  | &#10003; |  | 2024 | https://arxiv.org/abs/2402.16901 | Metagenomic pre-trained model for functional prediction. |
| DNABERT2 | &#10003; |  | &#10003; |  | &#10003; |  | 2023 | https://github.com/MAGICS-LAB/DNABERT_2 | Genomic pre-trained model trained on multi-species genomes. |
| HyenaDNA | &#10003; |  | &#10003; |  | &#10003; |  | 2023 | https://github.com/HazyResearch/hyena-dna | Genomic pre-trained model trained over human reference genome. |
| Nucleotide Transformer | &#10003; |  | &#10003; |  | &#10003; |  | 2023 | https://github.com/instadeepai/nucleotide-transformer | Genomic pre-trained model trained on diverse human genomes. |
| CNN-MGP |  | &#10003; |  | &#10003; | &#10003; |  | 2019 | https://github.com/rachidelfermi/cnn-mgp | Gene prediction using CNN network. |
| PlasGUN |  | &#10003; |  | &#10003; | &#10003; |  | 2020 | https://github.com/zhenchengfang/PlasGUN | Gene prediction tool using multiple CNN network. |
| PLM-ARG | &#10003; |  |  | &#10003; | &#10003; |  | 2023 | https://github.com/Junwu302/PLM-ARG | ARG identification framework using a pretrained protein language model. |
| DeepARG |  | &#10003; |  | &#10003; | &#10003; |  | 2018 | https://github.com/gaarangoa/deeparg | ARG prediction software by alignment and metagenomic sequences. |
| RGI |  | &#10003; |  | &#10003; | &#10003; |  | 2023 | https://github.com/arpcard/rgi | ARG prediction tools for annotating genes from scratch. |
| DeepVirFinder |  | &#10003; |  | &#10003; | &#10003; |  | 2020 | https://github.com/jessieren/DeepVirFinder | Viral sequences prediction with reference and alignment-free CNNs. |
| ViBE | &#10003; |  |  | &#10003; | &#10003; |  | 2022 | https://github.com/DMnBI/ViBE | Eukaryotic viruses identification with hierarchical BERT model. |
| ViraMiner |  | &#10003; |  | &#10003; | &#10003; |  | 2019 | https://github.com/NeuroCSUT/ViraMiner | Viral genomes identification in human samples. |
| DeepVF |  | &#10003; |  | &#10003; | &#10003; |  | 2021 | http://deepvf.erc.monash.edu/ | Viral factor identification with hybrid framework using stacking strategy. |
| HyperVR |  | &#10003; |  | &#10003; | &#10003; |  | 2023 | https://github.com/jiboyalab/HyperVR | Viral factors and mixing of ARG simultaneous prediction. |
| CLEAN |  | &#10003; |  | &#10003; | &#10003; |  | 2023 | https://github.com/tttianhao/CLEAN | Enzyme function prediction using contrastive learning. |
| DeepMicrobes |  | &#10003; |  | &#10003; | &#10003; |  | 2020 | https://github.com/MicrobeLab/DeepMicrobes | Taxonomic classification for metagenomics with self-attention model. |
| PopPhy-CNN |  | &#10003; |  | &#10003; |  | &#10003; | 2020 | https://github.com/YDaiLab/PopPhy-CNN | Host Phenotypes prediction by systematic tree embedded CNN network. |



## Visualization

We present visualization comparison of different methods on each ARG category of NCRD dataset below. For more detailed information, please refer to the notebook [plot_heatmap_fullmetric.ipynb](./OpenMeta/datasets/NCRD/heatmap_plot/plot_heatmap_fullmetric.ipynb) with full four metrics. 

<p align="center">
  <img src="./src/figure/Resistance_performance_heatmap_fullmetric.png" /> 
</p>

And the notebook [plot_heatmap.ipynb](./OpenMeta/datasets/NCRD/heatmap_plot/plot_heatmap.ipynb) shows the precision and F1-score metrics.

## Results
Below are results for algorithms trained on each dataset in OpenMeta. 

- Small-Scale Classification
- E-K12

| Method | E-K12 |
| --- | --- |
| RF (3-mer) | 20.2 |
| SVM (3-mer) | 38.6 |
| AdaBoost (3-mer) | 39.9 |
| LSTM (w2v) | 40.4 |
| LSTM (one-hot) | 38.1 |
| BiLSTM (w2v) | 40 |
| BiLSTM (one-hot) | 40.1 |
| BiLSTM-Att. (w2v) | 38.2 |
| BiLSTM-Att. (one-hot) | 40.8 |
| VT | 43.3 |
| HyenaDNA | 42.4 |
| NT | 45.1 |
| DNABert2 | 51.7 |
| FGBERT | 61.8 |

- CARD

| Method | CARD-A | CARD-D | CARD-R |
| --- | --- | --- | --- |
| RF (3-mer) | 22.4 | 36.1 | 47.8 |
| SVM (3-mer) | 27.6 | 33.6 | 43.3 |
| AdaBoost (3-mer) | 36.9 | 36.4 | 36.2 |
| LSTM (w2v) | 47.1 | 37.5 | 47.5 |
| LSTM (one-hot) | 46.2 | 39.1 | 41.5 |
| BiLSTM (w2v) | 43.3 | 35.5 | 36.3 |
| BiLSTM (one-hot) | 47.4 | 38.9 | 58.9 |
| BiLSTM-Att. (w2v) | 31.9 | 43.5 | 35.1 |
| BiLSTM-Att. (one-hot) | 46.7 | 31.2 | 41.6 |
| VT | 57.1 | 49.8 | 55.7 |
| HyenaDNA | 50.9 | 53.6 | 66.2 |
| NT | 58.5 | 56.2 | 68 |
| DNABert2 | 65.2 | 51.5 | 61.2 |
| FGBERT | 78.6 | 57.4 | 69.4 |
| DeepARG | - | 52.2 | 65.3 |
| PLM-ARG | - | - | 68.1 |
| RGI | - | - | - |

- PATRIC core dataset

| Method | PATRIC |
| --- | --- |
| RF (3-mer) | 25.3 |
| SVM (3-mer) | 26.6 |
| AdaBoost (3-mer) | 11 |
| LSTM (w2v) | 41.3 |
| LSTM (one-hot) | 19.9 |
| BiLSTM (w2v) | 20.3 |
| BiLSTM (one-hot) | 43.3 |
| BiLSTM-Att. (w2v) | 25.8 |
| BiLSTM-Att. (one-hot) | 18.5 |
| VT | 49.8 |
| HyenaDNA | 51.1 |
| NT | 46.1 |
| DNABert2 | 52.9 |
| FGBERT | 99.3 |


- Large-Scale Classification
- ENZYME

| Method | ENZYME |
| --- | --- |
| RF (3-mer) | 33.6 |
| SVM (3-mer) | 31.3 |
| AdaBoost (3-mer) | 31.4 |
| LSTM (w2v) | 42.8 |
| LSTM (one-hot) | 34.1 |
| BiLSTM (w2v) | 38.7 |
| BiLSTM (one-hot) | 31.6 |
| BiLSTM-Att. (w2v) | 36.9 |
| BiLSTM-Att. (one-hot) | 43.6 |
| VT | 68.2 |
| HyenaDNA | 79.6 |
| NT | 74.1 |
| DNABert2 | 85.4 |
| FGBERT | 99.1 |
| CLEAN | 92.3 |


- VFDB

| Method | VFDB |
| --- | --- |
| RF (3-mer) | 22.4 |
| SVM (3-mer) | 28 |
| AdaBoost (3-mer) | 27.3 |
| LSTM (w2v) | 36.7 |
| LSTM (one-hot) | 32.9 |
| BiLSTM (w2v) | 46.1 |
| BiLSTM (one-hot) | 31.3 |
| BiLSTM-Att. (w2v) | 37.7 |
| BiLSTM-Att. (one-hot) | 36.7 |
| VT | 58 |
| HyenaDNA | 61.1 |
| NT | 58.3 |
| DNABert2 | 58.2 |
| FGBERT | 75.7 |
| ViBE | 50.9 |

- NCycDB

| Method | NCycDB |
| --- | --- |
| RF (3-mer) | 67 |
| SVM (3-mer) | 66.9 |
| AdaBoost (3-mer) | 68.8 |
| LSTM (w2v) | 71.9 |
| LSTM (one-hot) | 65 |
| BiLSTM (w2v) | 66.9 |
| BiLSTM (one-hot) | 82 |
| BiLSTM-Att. (w2v) | 69 |
| BiLSTM-Att. (one-hot) | 67.3 |
| VT | 84.5 |
| HyenaDNA | 92.4 |
| NT | 75.1 |
| DNABert2 | 88.6 |
| FGBERT | 99.5 |


- Fine-Grained Classification
- NCRD

| Method | NCRD-Gene Family (420) |  |  |  |  | NCRD-Gene Name (1900) |  |  |  |  | NCRD-Resistance (30) |  |  |  |  | NCRD-Mechanisms (11) |  |  |  |  |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | Acc. | Pre. | Re. | F1 | FNR | Acc. | Pre. | Re. | F1 | FNR | Acc. | Pre. | Re. | F1 | FNR | Acc. | Pre. | Re. | F1 | FNR |
| DeepARG | - | - | - | - | - | 0.63 | 1.00 | 0.43 | 0.60 | 0.57 | 0.97 | 0.99 | 0.51 | 0.65 | 0.42 | - | - | - | - | - |
| RGI | 0.56 | 0.97 | 0.36 | 0.50 | 0.61 | 0.50 | 1.00 | 0.26 | 0.42 | 0.74 | 0.56 | 1.00 | 0.36 | 0.52 | 0.64 | 0.61 | 0.62 | 0.59 | 0.57 | 0.51 |
| PLM-ARG | - | - | - | - | - | - | - | - | - | - | 0.96 | 0.87 | 0.74 | 0.88 | 12.38 | - | - | - | - | - |
| FGBERT | 0.94 | 0.96 | 0.94 | 0.93 | 0.03 | 0.93 | 0.93 | 0.92 | 0.93 | 0.05 | 0.99 | 0.99 | 0.99 | 0.99 | 0.31 | 0.99 | 0.99 | 0.98 | 0.990.98 | 0.006 |

- Hierarchical Classification
- Cirrhosis

| Method | M.F1 | AUC | MCC | Pre. | Re. |
| :--------- | :------- | :------ | :------ | :------- | :------ |
| RF         | 0\.79    | 0\.93   | 0\.61   | 0\.88    | 0\.87   |
| SVM        | 0\.77    | 0\.89   | 0\.57   | 0\.85    | 0\.84   |
| AdaBoost   | 0\.70    | 0\.71   | 0\.43   | 0\.72    | 0\.71   |
| CNN        | 0\.84    | 0\.89   | 0\.68   | 0\.81    | 0\.80   |
| PopPhy     | 0\.81    | 0\.90   | 0\.61   | 0\.83    | 0\.82   |

- T2D

| Method | M.F1 | AUC | MCC | Pre. | Re. |
| :--------- | :------- | :------ | :------ | :------- | :------ |
| RF         | 0\.66    | 0\.72   | 0\.33   | 0\.67    | 0\.67   |
| SVM        | 0\.61    | 0\.63   | 0\.23   | 0\.61    | 0\.61   |
| AdaBoost   | 0\.70    | 0\.70   | 0\.42   | 0\.70    | 0\.70   |
| CNN        | 0\.59    | 0\.65   | 0\.19   | 0\.60    | 0\.59   |
| PopPhy     | 0\.58    | 0\.64   | 0\.18   | 0\.59    | 0\.58   |
