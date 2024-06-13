import os
GPU_NUMBER = [0,1,2,3]
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(s) for s in GPU_NUMBER])

import pandas as pd
import sys
from Bio import SeqIO
import subprocess

new_data = pd.DataFrame(columns=["contigs", "MGYP_proteins"])
fasta_path = sys.argv[1]
new_meta_file_path = sys.argv[2]
output_dir = os.path.dirname(new_meta_file_path)
subprocess.call(f"mkdir -p {output_dir}", shell=True)

print(fasta_path)
with open(fasta_path,'r') as fa:
    MGYPs = []
    count = 0
    for record in SeqIO.parse(fa,'fasta'):        
        MGYPs.append(record.id)        
        if len(MGYPs) == 30:
            contigs = 'contig'+str(count+1)
            MGYP_names = ";".join(MGYPs)
            new_data.loc[count] = [contigs, MGYP_names]
            count += 1
            MGYPs = []      
    if 15 < len(MGYPs) < 30 :
        contigs = 'contig'+str(count+1)
        MGYP_names = ";".join(MGYPs)
        new_data.loc[count] = [contigs, MGYP_names]
        count += 1
        MGYPs = []
new_meta_file_dir = new_meta_file_path.split('/')
new_data.to_csv(new_meta_file_path, header=None, index=False)