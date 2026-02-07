import sys
import subprocess
from CLEAN.utils import mutate_single_seq_ECs, retrive_esm1b_embedding, compute_esm_distance
from CLEAN.utils import *

def determine_epoch(train_file):
    epoch_mapping = {
        "split10": "2000",
        "split30": "2500",
        "split50": "3500",
        "split70": "5000",
        "split100": "7000"
    }
    return epoch_mapping.get(train_file, "2000")

def determine_supconH_epoch(train_file):
    triplet_epoch = int(determine_epoch(train_file))
    supconH_epoch = int(triplet_epoch * 0.75)
    return str(supconH_epoch)

def main(train_file):
    # Task 1: Mutate single sequences and retrieve ESM1b embedding
    print('mutate_single_seq_ECs')
    train_fasta_file = mutate_single_seq_ECs(train_file)
    
    print('retrive_esm1b_embedding')
    # csv_to_fasta("data/split10.csv", "data/split10.fasta")
    retrive_esm1b_embedding(train_file)
    retrive_esm1b_embedding(train_fasta_file)

    # Task 2: Compute ESM distance
    print('compute_esm_distance')
    compute_esm_distance(train_file)

    # Task 3: Train triplet model with determined epoch
    print(train_file, model_name)
    epoch = determine_epoch(train_file)
    subprocess.run(["python", "./train-triplet.py", "--training_data", train_file, 
                    "--model_name", train_file + "_triplet", "--epoch", epoch])
    
    # python ./train-triplet.py --training_data split10 --model_name split10_triplet --epoch 2500

    # Task 4: Train supconH model To train a CLEAN model with SupCon-Hard loss, and take 10% split as an example, run:
    supconH_epoch = determine_supconH_epoch(train_file)
    subprocess.run(["python", "./train-supconH.py", "--training_data", train_file, 
                    "--model_name", train_file + "_supconH", "--epoch", supconH_epoch, 
                    "--n_pos", "9", "--n_neg", "30", "-T", "0.1"])
    

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python run_tasks.py <train_file>")
        sys.exit(1)

    train_file = sys.argv[1]
    main(train_file)
