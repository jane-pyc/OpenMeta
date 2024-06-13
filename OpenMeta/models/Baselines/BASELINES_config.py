class Config:
    def __init__(self):
        self.patho_path = ''
        self.nonpatho_path = ''
        self.nums_dic = {3: 32, 4: 136, 5: 512, 6: 2080, 7: 8192, 'all': 10952}
        self.freqs_nums = self.nums_dic['all']
        self.hidden_layers = [512, 256, 128]
        self.deep_layers = [4096, 2048, 1024, 512]
        # set CrossNet layers
        self.num_cross_layers = 7
        self.end_dims = [1024, 512, 256]
        self.out_layer_dims = 1024
        self.val_size = 0.2
        self.fold = 5
        self.test_size = 0.5
        self.random_state = 1
        self.num_epoch = 200
        self.patience = 30
        self.batch_size = 512
        self.Dropout = 0.3
        self.lr = 0.0000002
        self.l2_regularization = 0.00001
        self.device_id = 0
        self.use_cuda = True
        self.save_model = True
        self.output_base_path = ''
        self.best_model_name = 'model.pt'
        self.raw_fasta_path = 'example_data'
        self.combined_fasta_path = '../../../src/datasets/Small-Scale/E-K12/E-K12.fasta'
        self.ks = [3]
        self.num_procs = 8
        self.freqs_file = 'temp_data/Gene_operons_3-mer.pkl'
        self.save_res_path = 'temp_data/results_for_pre_Gene_sequence_3-mer.csv'
        self.new_meta_file_path = 'temp_data/new_meta_file_3-mer.csv'
        self.batch_data_path = 'temp_data/batch_data_3-mer.pkl'
        self.batch_data_prot_index_dict_path = 'temp_data/batch_data_3-mer_prot_index_dict.pkl'
        self.annot_path = '../../../src/datasets/Small-Scale/E-K12/E-K12.annot'
        self.pkl_path = 'temp_data/batch_data_3-mer.pkl'
        
        
