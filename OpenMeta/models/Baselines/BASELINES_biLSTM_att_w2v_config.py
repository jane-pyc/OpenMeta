class Config:
    def __init__(self):
        self.raw_fasta_path = 'example_data'
        self.combined_fasta_path = '../../../src/datasets/Small-Scale/E-K12/E-K12.fasta'
        #  settings for calculation of kmer frequency of fasta
        # self.ks = [3, 4, 5, 6, 7]
        self.ks = [3]
        self.num_procs = 8
        self.freqs_file = 'temp_data/Gene_operons_biLSTM_att_w2v.pkl'
        self.save_res_path = 'temp_data/results_for_pre_Gene_sequence_3-mer.csv'
        self.new_meta_file_path = 'temp_data/new_meta_file_biLSTM_att_w2v.csv'
        self.batch_data_path = 'temp_data/batch_data_biLSTM_w2v-960.pkl'
        self.batch_data_prot_index_dict_path = 'temp_data/batch_data_biLSTM_att_w2v_prot_index_dict.pkl'
        self.annot_path = '../../../src/datasets/Small-Scale/E-K12/E-K12.annot'
        self.pkl_path = 'temp_data/batch_data_w2v-960.pkl'      
        
