# BASELINES Pipeline

## Overview
This repository documents the BASELINES computational pipeline designed for gene operon analysis via k-mer frequency calculations and machine learning classification. The workflow integrates sequence processing, data formatting, and predictive modeling, culminating in operon interaction predictions through robust statistical evaluation.

## Configuration Settings (`BASELINES_config.py`)
- **Combined FASTA Path:** `self.combined_fasta_path = '../../../src/datasets/Small-Scale/E-K12/E-K12.fasta'`
- **K-mer Frequency Calculation:**
  - Selected k-mers: `self.ks = [3]`
  - Number of processes: `self.num_procs = 8`
  - Frequency file output: `self.freqs_file = 'temp_data/Gene_operons_3-mer.pkl'`
  - Results file: `self.save_res_path = 'temp_data/results_for_pre_Gene_sequence_3-mer.csv'`
- **K-mer Data Pre-processing:**
  - Metadata file: `self.new_meta_file_path = 'temp_data/new_meta_file_3-mer.csv'`
  - Batch data files:
    - `self.batch_data_path = 'temp_data/batch_data_3-mer.pkl'`
    - `self.batch_data_prot_index_dict_path = 'temp_data/batch_data_3-mer_prot_index_dict.pkl'`
- **Machine Learning Model Configuration:**
  - Annotation path: `self.annot_path = '../../../src/datasets/Small-Scale/E-K12/E-K12.annot'`
  - Pickle path for batch data: `self.pkl_path = 'temp_data/batch_data_3-mer.pkl'`

## Pipeline Steps (`BASELINES_predict.py`)
### Step 1: Calculate k-mer Frequencies
- **Function:** `cal_main`
- **Inputs:** Operon fasta file.
- **Outputs:** A pickle file containing 3-mer representations.
- **Format:**
  - Input FASTA: Header line (`>`) followed by base sequence.
  - Output 3-mer format: List of tuples, each containing operon name and 32D representation.

### Step 2: Sequence Metadata Processing
- **Function:** `seq_meta_data_process`
- **Inputs:** Operon fasta file.
- **Outputs:** Contig.tsv table grouping 30 genes per entry.
- **Formats:**
  - Input: `config.combaiine_fasta_path`
  - Output: `config.new_meta_file_path`

### Step 3: Batch Data Compilation
- **Function:** `batch_data`
- **Inputs:** 3-mer pickle file and contig.tsv table.
- **Outputs:** Batch data pickle file.
- **Data structure:** Includes 'prot_ids' and 'embeds'.
- **Formats:**
  - Input files: `config.freqs_file`, `config.new_meta_file_path`
  - Output file: `config.batch_data_path`

## Prediction and Validation
- **Objective:** Predict whether gene pairs belong to the same operon based on their 32D vector representation.
- **Methodology:** Utilize machine learning classification with 5-fold cross-validation to compute AUC and ROC metrics.
- **Details:** For each gene pair:
  - Vector (`X.append`): 32D-vector from the interaction data.
  - Label (`y.append`): 1 if same operon, 0 otherwise.
