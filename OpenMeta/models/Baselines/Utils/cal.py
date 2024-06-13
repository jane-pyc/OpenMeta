import itertools
import multiprocessing as mp
import time
from multiprocessing import Manager
from Bio import SeqIO
from gensim.models import Word2Vec
import torch

import numpy as np
import pickle as pkl
from Utils.cal_utils import count_kmers, readfq, mer2bits, get_rc, count_kmers_biLSTM_att_w2v


def compute_kmer_inds(ks):
    ''' Get the indeces of each canonical kmer in the kmer count vectors
    '''
    kmer_list = []
    kmer_inds = {k: {} for k in ks}
    kmer_count_lens = {k: 0 for k in ks} #不重复k-mer的数量

    alphabet = 'ACGT'
    for k in ks:
        all_kmers = [''.join(kmer) for kmer in itertools.product(alphabet, repeat=k)] #len=4**k
        # 生成所有kmers的组合 4的k次方个          
        all_kmers.sort()
        # 排序 以ascii码大小排
        ind = 0
        for kmer in all_kmers:
            bit_mer = mer2bits(kmer)
            rc_bit_mer = mer2bits(get_rc(kmer)) # Return the reverse complement of seq
            if rc_bit_mer in kmer_inds[k]: # len(kmer_inds[k])=4**k
                kmer_inds[k][bit_mer] = kmer_inds[k][rc_bit_mer]
            else:
                kmer_list.append(kmer)
                kmer_inds[k][bit_mer] = ind
                kmer_count_lens[k] += 1
                ind += 1
    # return kmer_inds, kmer_count_lens, kmer_list
    print("finishing computing kmer inds")
    return kmer_inds, kmer_count_lens


def get_seq_lengths(infile):
    ''' Read in all the fasta entries,
        return arrays of the headers, and sequence lengths
    '''
    sequence_names = []
    sequence_lengths = []
    seqs = []
    fp = open(infile)
    for name, seq, _ in readfq(fp):
        sequence_names.append(name)
        sequence_lengths.append(len(seq))
        seqs.append(seq)
    fp.close()
    # print('len of sequence_lengths', len(sequence_names), len(sequence_lengths))
    return sequence_names, sequence_lengths, seqs


def get_seq(infile):
    seq_names = []
    seqs = []
    fp = open(infile)
    i = 0
    for name, seq, _ in readfq(fp):
        seq_names.append(name)
        seqs.append(seq)
        # if len(seqs) == 63911:
        #     break
        i += 1
        # if i % 100000 == 0:
        # if i < 100000:
    print("Read {} sequences".format(i))
    return seq_names, seqs



def get_seqs(infile):
    ''' Create array of the sequences
    '''
    seqs = []
    fp = open(infile)
    for name, seq, _ in readfq(fp):
        seqs.append(seq)

    # start_inds = inds_dict.get(name, [])
    # for start_ind in start_inds:
    #     frag = seq[start_ind:start_ind+l]
    #     # if len(frag) < l and len(seq) > l:
    #     #     frag += seq[:l-len(frag)]
    #     seqs.append(frag)
    fp.close()
    return seqs


def cal_kmer_freqs(seq_file, num_procs, ks):
    # ks=[3]
    names, seqs = get_seq(seq_file)
    names_seqs = list(zip(names,seqs))

    time_start = time.time()
    # patho_names, patho_lengths, patho_seqs = get_seq_lengths(patho_file)
    ## for l in lens:
    # coverage=1 # TODO: make this command line option
    kmer_inds, kmer_count_lens = compute_kmer_inds(ks)
    # print('kmer_inds: /n',kmer_inds)
    # print('kmer_count_lens:/n', kmer_count_lens)
    pool = mp.Pool(num_procs)
    # 
    patho_list = Manager().list()
    for cur in np.arange(len(seqs)):
        patho_list.append((str(cur),0))
    
    pool.map(count_kmers, [[ind, name_seq, ks, kmer_inds, kmer_count_lens, patho_list] \
                           for ind, name_seq in enumerate(names_seqs)])

    # patho_freqs = np.array(patho_list) #old:(data nums,32)
    patho_freqs_list = list(patho_list)
    pool.close()
    time_end = time.time()
    print('costs:', int(time_end - time_start), 's')

    return names, patho_freqs_list


def cal_main(combined_fasta_path, num_procs, ks, freqs_file):
    names, freqs = cal_kmer_freqs(combined_fasta_path, num_procs, ks)
    # np.save(freqs_file, freqs)
    # pkl.dump()
    with open(freqs_file,'wb') as f:
        pkl.dump(freqs, f)
    return names


def biLSTM_w2v(combined_fasta_path, num_procs, freqs_file, max_sequence_length):
    seq_file = combined_fasta_path
    names, seqs = get_seq(seq_file) #len=4315
       
    # 读取fasta格式的氨基酸序列数据
    fasta_file = combined_fasta_path
    sequences = [str(record.seq) for record in SeqIO.parse(fasta_file, "fasta")] #len=4315

    # 将氨基酸序列转换为词嵌入（Word2Vec）编码
    embedding_dim = 100  # 设置词嵌入维度，根据你的需求调整
    word2vec_model = Word2Vec(sentences=[list(seq) for seq in sequences], vector_size=embedding_dim, window=5, min_count=1, sg=0)

    # 创建一个词汇表，将氨基酸映射到Word2Vec嵌入向量
    vocab = word2vec_model.wv.key_to_index

    # 将氨基酸序列编码成Word2Vec嵌入向量
    encoded_sequences = []
    for seq in sequences:
        encoded_seq = [vocab[aa] for aa in seq if aa in vocab]
        encoded_sequences.append(encoded_seq)

    max_sequence_length = max_sequence_length
    padded_sequences = []
    for seq in encoded_sequences:
        if len(seq) > max_sequence_length:
            # 如果序列超过最大长度，截断到最大长度
            padded_seq = seq[:max_sequence_length]
        else:
            pad_length = max_sequence_length - len(seq)
            padded_seq = seq + [0] * pad_length  # 使用0填充
        padded_sequences.append(padded_seq)

    # 将编码后的序列转换为PyTorch张量
    encoded_sequences_tensor = torch.tensor(padded_sequences, dtype=torch.long) #[4315, 128])
    
    names_w2v = list(zip(names,encoded_sequences_tensor))    
    pool = mp.Pool(num_procs)
    patho_list = Manager().list()
    for cur in np.arange(len(seqs)):
        patho_list.append((str(cur),0))
    
    pool.map(count_kmers_biLSTM_att_w2v, [[ind, names_w2v, patho_list] \
                           for ind, names_w2v in enumerate(names_w2v)])

    # patho_freqs = np.array(patho_list)
    patho_freqs_list = list(patho_list)
    pool.close()
    
    freqs = patho_freqs_list #[('thrL', tensor([2, 3, 0, 2, ... 0, 0, 0])), 128-D
    with open(freqs_file,'wb') as f:
        pkl.dump(freqs, f)
    return names


def biLSTM_onehot(combined_fasta_path, num_procs, freqs_file, max_sequence_length, cut_out_value):
    seq_file = combined_fasta_path
    names, seqs = get_seq(seq_file) #len=4315
       
    # 读取fasta格式的氨基酸序列数据
    fasta_file = combined_fasta_path
    sequences = [str(record.seq) for record in SeqIO.parse(fasta_file, "fasta")] #len=4315

    # 确定最大序列长度
    max_sequence_length = max(len(seq) for seq in sequences)
    cut_out_value = cut_out_value

    # 定义氨基酸字典，将氨基酸映射到整数
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    amino_acid_dict = {aa: i for i, aa in enumerate(amino_acids)}

    # 将氨基酸序列转换为One-Hot编码并进行填充
    def sequence_to_one_hot(sequence, amino_acid_dict, max_length):
        one_hot_sequence = np.zeros((max_length, len(amino_acid_dict)), dtype=np.float32)
        for i, aa in enumerate(sequence):
            if aa in amino_acid_dict:
                one_hot_sequence[i, amino_acid_dict[aa]] = 1.0
        return one_hot_sequence

    # 将所有氨基酸序列转换为One-Hot编码并进行填充
    one_hot_sequences = [sequence_to_one_hot(seq, amino_acid_dict, max_sequence_length) for seq in sequences]

    # 截取序列，只保留前 cut_out_value 个位置
    one_hot_sequences = np.array(one_hot_sequences)[:, :cut_out_value, :]

    # 将One-Hot编码数据转换为PyTorch张量
    input_data = torch.tensor(one_hot_sequences, dtype=torch.float32)  #([4315, 32, 20]) 4315 条输入序列，每个序列维度是32（cuo off value）,20个aa字符

    # 将编码后的序列转换为PyTorch张量
    encoded_sequences_tensor = input_data #[4315, 32, 20])
    
    names_w2v = list(zip(names,encoded_sequences_tensor))    
    pool = mp.Pool(num_procs)
    patho_list = Manager().list()
    for cur in np.arange(len(seqs)):
        patho_list.append((str(cur),0))
    
    pool.map(count_kmers_biLSTM_onehot, [[ind, names_w2v, patho_list] \
                           for ind, names_w2v in enumerate(names_w2v)])

    # patho_freqs = np.array(patho_list)
    patho_freqs_list = list(patho_list)
    pool.close()
    
    freqs = patho_freqs_list #[('thrL', tensor([2, 3, 0, 2, ... 0, 0, 0])), [32,20]-D
    with open(freqs_file,'wb') as f:
        pkl.dump(freqs, f)
    return names


def biLSTM_att_w2v(combined_fasta_path, num_procs, freqs_file):
    seq_file = combined_fasta_path
    names, seqs = get_seq(seq_file) #len=4315
       
    # 读取fasta格式的氨基酸序列数据
    fasta_file = combined_fasta_path
    sequences = [str(record.seq) for record in SeqIO.parse(fasta_file, "fasta")] #len=4315

    # 将氨基酸序列转换为词嵌入（Word2Vec）编码
    embedding_dim = 100  # 设置词嵌入维度，根据你的需求调整
    word2vec_model = Word2Vec(sentences=[list(seq) for seq in sequences], vector_size=embedding_dim, window=5, min_count=1, sg=0)

    # 创建一个词汇表，将氨基酸映射到Word2Vec嵌入向量
    vocab = word2vec_model.wv.key_to_index

    # 将氨基酸序列编码成Word2Vec嵌入向量
    encoded_sequences = []
    for seq in sequences:
        encoded_seq = [vocab[aa] for aa in seq if aa in vocab]
        encoded_sequences.append(encoded_seq)

    # 获取每个序列的实际长度
    # sequence_lengths = [len(seq) for seq in encoded_sequences]

    # 对不等长的序列进行填充，以使它们具有相同的长度
    # 这是为了使它们适用于BiLSTM-Attention的批处理
    # max_sequence_length = max(sequence_lengths)
    max_sequence_length = 10
    padded_sequences = []
    for seq in encoded_sequences:
        if len(seq) > max_sequence_length:
            # 如果序列超过最大长度，截断到最大长度
            padded_seq = seq[:max_sequence_length]
        else:
            pad_length = max_sequence_length - len(seq)
            padded_seq = seq + [0] * pad_length  # 使用0填充
        padded_sequences.append(padded_seq)

    # 将编码后的序列转换为PyTorch张量
    encoded_sequences_tensor = torch.tensor(padded_sequences, dtype=torch.long) #[4315, 150])
    
    names_w2v = list(zip(names,encoded_sequences_tensor))    
    pool = mp.Pool(num_procs)
    patho_list = Manager().list()
    for cur in np.arange(len(seqs)):
        patho_list.append((str(cur),0))
    
    pool.map(count_kmers_biLSTM_att_w2v, [[ind, names_w2v, patho_list] \
                           for ind, names_w2v in enumerate(names_w2v)])

    # patho_freqs = np.array(patho_list)
    patho_freqs_list = list(patho_list)
    pool.close()
    
    freqs = patho_freqs_list #[('thrL', tensor([2, 3, 0, 2, ... 0, 0, 0])), 150-D
    with open(freqs_file,'wb') as f:
        pkl.dump(freqs, f)
    return names