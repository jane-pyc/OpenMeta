class Config:
    def __init__(self):
        self.raw_fasta_path = 'example_data'
        self.combined_fasta_path = '../../../src/datasets/Large-Scale/NCycDB/NCyc.fasta'
        self.ks = [3]
        self.num_procs = 2
        self.freqs_file = 'temp_data_NCyc/biLSTM_att_w2v.pkl'
        self.save_res_path = 'temp_data_NCyc/results_for_pre_Gene_sequence_3-mer2.csv'
        self.new_meta_file_path = 'temp_data_NCyc/new_meta_file_biLSTM_att_w2v.csv'
        self.batch_data_path = 'temp_data_NCyc/batch_data_biLSTM_w2v-960.pkl'
        self.batch_data_prot_index_dict_path = 'temp_data_NCyc/batch_data_biLSTM_att_w2v_prot_index_dict.pkl'
        self.annot_path = ''
        self.pkl_path = 'temp_data_NCyc/batch_data_w2v-960.pkl'
        
        
