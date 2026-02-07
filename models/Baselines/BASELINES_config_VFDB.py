class Config:
    def __init__(self):
        self.combined_fasta_path = '../../../src/datasets/Large-Scale/VFDB/VFDB.fasta'
        self.ks = [3]
        self.num_procs = 8
        self.freqs_file = 'temp_data/3-mer.pkl'
        self.save_res_path = 'temp_data/results_for_pre_Gene_sequence_3-mer.csv'
        self.new_meta_file_path = 'temp_data/new_meta_file_3-mer.csv'
        self.batch_data_path = 'temp_data/batch_data_3-mer.pkl'
        self.batch_data_prot_index_dict_path = 'temp_data/batch_data_3-mer_prot_index_dict.pkl'
        self.annot_path = ''
        self.pkl_path = 'temp_data/batch_data_3-mer.pkl'
        
        
