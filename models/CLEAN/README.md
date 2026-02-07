# Benchmark Bioinformatics Workflow

## Overview
This repository details the processes involved in generating mutations, embedding sequences, and training a model using ESM-1b representations. This workflow is crucial for understanding protein function through computational means, including sequence labeling, mutation simulation, and model training for classification or regression.

## Data Generation Process

### 1. Initial Data Setup
- **File:** `train_file = "split10.csv"` (Original CSV data containing EC number labels and sequences)
- **Location:** Stored at `/data/train/`

### 2. Mutation Simulation
- **Function Used:** `mutate_single_seq_ECs`
- **Output File:** 
  - Generated with the command: `output_fasta = open('./data/' + fasta_name + '.fasta','w')`
  - `fasta_name` is determined by appending '_single_seq_ECs' to `train_file`
- **Output Format:**
  - The fasta file includes sequences with masked regions, indicated by `<mask>`.
  - Example entries:
    ```
    >Q898E8_0
    METIKITTA<mask>ALIKFLNQQYVELDGQEYR<mask>VQGIFTIFGHG<mask>VLGIGQA<mask>EEDPGHLEVYQGHNE<mask>GMAQ<mask>AIAFAKQSNR<mask>QIY<mask>...
    ```

### 3. ESM-1b Representation Retrieval
- **Function:** `retrieve_esm1b_embedding`

### 4. Euclidean Distance Computation
- **Function:** `compute_esm_distance`
- **Outputs:**
  - Saves the distance matrix and ESM embedding matrix as `split10.pkl` and `split10_esm.pkl` in `/data/distance_map`.

### 5. Model Training
- **Command:**
  ```bash
  python ./train-triplet.py --training_data split10 --model_name split10_triplet --epoch 2500


  python run_train_clean.py split100

Model Weights Storage: /data/model/split10_triplet.pth
Recommended Epochs by Data Split:
10% split: 2000 epochs
30% split: 2500 epochs
50% split: 3500 epochs
70% split: 5000 epochs
100% split: 7000 epochs