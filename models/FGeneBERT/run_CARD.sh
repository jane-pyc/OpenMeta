python plm_embed.py \
    --fasta ../../../src/datasets/Small-Scale/CARD-A/CARD.fasta \
    --out ./Data_process/CARD_seq_cate_fasta.esm.embs.pkl \
    --port 23441

python seq_meta_data_process.py \
    --fasta ../../../src/datasets/Small-Scale/CARD-A/CARD.fasta \
    --out_tsv ./Data_process/CARD_seq_cate_fasta.tsv

python batch_data.py \
    --emb_pkl ./Data_process/CARD_seq_cate_fasta.esm.embs.pkl \
    --meta_tsv ./Data_process/CARD_seq_cate_fasta.tsv \
    --out_dir ./Data_process/batched_data
