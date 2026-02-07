class Config:
    def __init__(self):
        self.raw_fasta_path = 'example_data'
        self.combined_fasta_path = '../../../src/datasets/Small-Scale/CARD-A/CARD.fasta'
        self.ks = [3]
        self.num_procs = 8
        self.freqs_file = 'temp_data_CARD/CARD_3-mer.pkl'
        self.save_res_path = 'temp_data_CARD/results_for_pre_Gene_sequence_3-mer.csv'
        self.new_meta_file_path = 'temp_data_CARD/new_meta_file_3-mer.csv'
        self.batch_data_path = 'temp_data_CARD/batch_data_3-mer.pkl'
        self.batch_data_prot_index_dict_path = 'temp_data_CARD/batch_data_3-mer_prot_index_dict.pkl'
        self.annot_path = ''
        self.pkl_path = 'temp_data_CARD/batch_data_3-mer.pkl'
        
        
