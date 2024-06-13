python plm_embed.py ../../../src/datasets/Large-Scale/ENZYME/ENZYME.fasta ./Data_process/ENZYME_seq_cate_fasta.esm.embs.pkl 23448

python seq_meta_data_process.py ../../../src/datasets/Large-Scale/ENZYME/ENZYME.fasta ./Data_process/ENZYME_seq_cate_fasta.tsv

python batch_data.py ./Data_process/ENZYME_seq_cate_fasta.esm.embs.pkl ./Data_process/ENZYME_seq_cate_fasta.tsv ./Data_process/batched_data

python glm_embed.py -d ./Data_process/batched_data -m ./model/pytorch_model.bin -o ./output --all_results --attention