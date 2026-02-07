import os
import argparse
import subprocess
import pandas as pd
from Bio import SeqIO

parser = argparse.ArgumentParser()
parser.add_argument('--fasta', type=str, required=True, help='Input fasta file')
parser.add_argument('--out_tsv', type=str, required=True, help='Output tsv file')
parser.add_argument('--window', type=int, default=30, help='Window size for grouping')
parser.add_argument('--min_size', type=int, default=15, help='Minimum group size')
args = parser.parse_args()

output_dir = os.path.dirname(args.out_tsv)
if output_dir:
    subprocess.call(f"mkdir -p {output_dir}", shell=True)

new_data = pd.DataFrame(columns=["contigs", "MGYP_proteins"])

print(f"Processing: {args.fasta}")
with open(args.fasta, 'r') as fa:
    buffer = []
    count = 0
    for record in SeqIO.parse(fa, 'fasta'):
        buffer.append(record.id)
        if len(buffer) == args.window:
            contigs = f'contig{count + 1}'
            names = ";".join(buffer)
            new_data.loc[count] = [contigs, names]
            count += 1
            buffer = []
    if args.min_size < len(buffer) < args.window:
        contigs = f'contig{count + 1}'
        names = ";".join(buffer)
        new_data.loc[count] = [contigs, names]
        count += 1

new_data.to_csv(args.out_tsv, header=None, index=False)
print(f"Output: {args.out_tsv}, total contigs: {count}")
