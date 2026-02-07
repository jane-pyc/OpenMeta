import os
GPU_NUMBER = [0,1,2,3,4,5,6,7]
os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(s) for s in GPU_NUMBER])

import csv
import random
import os
import math
from re import L
import torch
import numpy as np
import subprocess
import pickle
from .distance_map import get_dist_map

def seed_everything(seed=1234):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def extract_rows(csv_path, prot_ids_tensor):
    # 将张量转换为一维列表，并去除重复项
    unique_ids = set(prot_ids_tensor.flatten().tolist())

    # 读取 CSV 文件并存储每行内容
    rows_dict = {}
    with open(csv_path, newline='', encoding='utf-8') as csvfile:
        csvreader = csv.reader(csvfile, delimiter='\t')
        next(csvreader)  # 跳过表头
        for i, row in enumerate(csvreader, start=1):
            rows_dict[i] = row

    # 重新组织行以匹配原始张量的形状 [272, 30]
    selected_rows = []
    for row_indices in prot_ids_tensor.tolist():
        current_rows = [rows_dict.get(i + 1) for i in row_indices]
        selected_rows.append(current_rows)

    return selected_rows

    # selected_rows
    # [[[...], [...], [...], [...], [...], [...], [...], [...], [...], ...], [[...], [...], [...], [...], [...], [...], [...], [...], [...], ...], [[...], [...], [...], [...], [...], [...], [...], [...], [...], ...], [[...], [...], [...], [...], [...], [...], [...], [...], [...], ...], [[...], [...], [...], [...], [...], [...], [...], [...], [...], ...], [[...], [...], [...], [...], [...], [...], [...], [...], [...], ...], [[...], [...], [...], [...], [...], [...], [...], [...], [...], ...], [[...], [...], [...], [...], [...], [...], [...], [...], [...], ...], [[...], [...], [...], [...], [...], [...], [...], [...], [...], ...], [[...], [...], [...], [...], [...], [...], [...], [...], [...], ...], [[...], [...], [...], [...], [...], [...], [...], [...], [...], ...], [[...], [...], [...], [...], [...], [...], [...], [...], [...], ...], [[...], [...], [...], [...], [...], [...], [...], [...], [...], ...], [[...], [...], [...], [...], [...], [...], [...], [...], [...], ...], ...]
    # len(selected_rows) = 272
    # len(selected_rows[0]) = 30
    # (selected_rows[0])
    # [['B6VQ60', '2.3.2.27', 'MADVALRITETVARLQKELK...AVDSIGCLQD'], ['Q8L649', '2.3.2.27', 'MNGDNRPVEDAHYTETGFPY...SEVFGEPSIH'], ['D7UER1', '1.14.14.155', 'MAMETGLIFHPYMRPGRSAR...VAPKILLPKR'], ['O87875', '1.3.7.8;1.3.99.n1', 'MSAKTNPEVIKESSMVKQKE...KRSGASLATA'], ['F1SPM8', '2.7.11.1', 'MKKFFDSRREQGGSGLGSGS...LLLVDQLIDL'], ['Q52424', '2.3.1.59', 'MGIEYRSLHTSQLTLSEKEA...YCDFRGGDQW'], ['P10051', '2.3.1.82', 'MNYQIVNIAECSNYQLEAAN...IWMWKSLIKE'], ['P19650', '2.3.1.82', 'MSIQHFQTKLGITKYSIVTN...QAFERTRSVA'], ['Q9RBW7', '2.3.1.82', 'MIASAPTIRQATPADAAAWA...YFRMPLEPSA'], ['P83252', '3.2.1.23', 'TGVTYDHRALVIDGXXXVLV...AASSWYAVET'], ['Q4H4F3', '3.5.1.112', 'MNQDKRAFMFISPHFDDVIL...RSLSPEPLQT'], ['Q4H4F0', '4.3.2.6', 'MISWTKAFTKPLKGRIFMPN...EEKQDVPHSL'], ['Q89H53', '3.5.4.2', 'MRQVMNAIPDDLLITAPDEV...QTGEAHPVIW'], ['P30812', '3.2.1.23', 'MRIIENFNEFGEWGMTWRDG...GTQRITFNVT'], ...]
    # (selected_rows[0][0])
    # ['B6VQ60', '2.3.2.27', 'MADVALRITETVARLQKELK...AVDSIGCLQD']

  
def get_ec_id_dict_by_rows(flattened_rows):
    id_ec = {}
    ec_id = {}

    for row in flattened_rows:
        if row:  # 确保行非空
            entry, ec_numbers, _ = row
            ec_list = ec_numbers.split(';')
            id_ec[entry] = ec_list
            for ec in ec_list:
                if ec not in ec_id:
                    ec_id[ec] = set()
                ec_id[ec].add(entry)

    return id_ec, ec_id

def get_ec_id_dict(csv_name: str) -> dict:
    csv_file = open(csv_name)
    csvreader = csv.reader(csv_file, delimiter='\t')
    id_ec = {}
    ec_id = {}

    for i, rows in enumerate(csvreader):
        if i > 0:
            id_ec[rows[0]] = rows[1].split(';')
            # 这行代码将每个样本ID（rows[0]）映射到一个或多个EC编号上。EC编号是通过分号分隔的字符串（rows[1]）得到的。
            
            for ec in rows[1].split(';'): # 每个EC编号映射到一个包含所有具有该EC编号的样本ID的集合。
                if ec not in ec_id.keys():
                    ec_id[ec] = set()
                    ec_id[ec].add(rows[0])
                else:
                    ec_id[ec].add(rows[0])
    
    # id_ec 字典包含了每个样本ID与其对应的一个或多个EC编号的关系，而 ec_id 字典包含了每个EC编号与其对应的所有样本ID的关系。
    return id_ec, ec_id

def get_ec_id_dict_non_prom(csv_name: str) -> dict:
    # 它只处理那些具有单一EC编号的样本。
    
    csv_file = open(csv_name)
    csvreader = csv.reader(csv_file, delimiter='\t')
    id_ec = {}
    ec_id = {}

    for i, rows in enumerate(csvreader):
        if i > 0:
            if len(rows[1].split(';')) == 1:
                id_ec[rows[0]] = rows[1].split(';')
                for ec in rows[1].split(';'):
                    if ec not in ec_id.keys():
                        ec_id[ec] = set()
                        ec_id[ec].add(rows[0])
                    else:
                        ec_id[ec].add(rows[0])
    return id_ec, ec_id


def format_esm(a):
    if type(a) == dict:
        a = a['mean_representations'][33]
    return a


def load_esm(lookup):
    esm = format_esm(torch.load('./data/esm_data/' + lookup + '.pt'))
    return esm.unsqueeze(0)


def esm_embedding(ec_id_dict, device, dtype):
    '''
    Loading esm embedding in the sequence of EC numbers
    prepare for calculating cluster center by EC
    '''
    esm_emb = []
    # for ec in tqdm(list(ec_id_dict.keys())):
    for ec in list(ec_id_dict.keys()):
        ids_for_query = list(ec_id_dict[ec])
        esm_to_cat = [load_esm(id) for id in ids_for_query]
        esm_emb = esm_emb + esm_to_cat
    return torch.cat(esm_emb).to(device=device, dtype=dtype)


def model_embedding_test(id_ec_test, model, device, dtype):
    '''
    Instead of loading esm embedding in the sequence of EC numbers
    the test embedding is loaded in the sequence of queries
    then inferenced with model to get model embedding
    '''
    ids_for_query = list(id_ec_test.keys())
    esm_to_cat = [load_esm(id) for id in ids_for_query]
    esm_emb = torch.cat(esm_to_cat).to(device=device, dtype=dtype)
    model_emb = model(esm_emb)
    return model_emb

def model_embedding_test_ensemble(id_ec_test, device, dtype):
    '''
    Instead of loading esm embedding in the sequence of EC numbers
    the test embedding is loaded in the sequence of queries
    '''
    ids_for_query = list(id_ec_test.keys())
    esm_to_cat = [load_esm(id) for id in ids_for_query]
    esm_emb = torch.cat(esm_to_cat).to(device=device, dtype=dtype)
    return esm_emb

def csv_to_fasta(csv_name, fasta_name):
    csvfile = open(csv_name, 'r')
    csvreader = csv.reader(csvfile, delimiter='\t')
    outfile = open(fasta_name, 'w')
    for i, rows in enumerate(csvreader):
        if i > 0:
            outfile.write('>' + rows[0] + '\n')
            outfile.write(rows[2] + '\n')
            
def ensure_dirs(path):
    if not os.path.exists(path):
        os.makedirs(path)
        
def retrive_esm1b_embedding(fasta_name):
    esm_script = "esm/scripts/extract.py"
    esm_out = "data/esm_data"
    esm_type = "esm1b_t33_650M_UR50S"
    fasta_name = "data/" + fasta_name + ".fasta"
    command = ["python", esm_script, esm_type, 
              fasta_name, esm_out, "--include", "mean"]
    subprocess.run(command)
 
def compute_esm_distance(train_file):
    ensure_dirs('./data/distance_map/')
    _, ec_id_dict = get_ec_id_dict('./data/' + train_file + '.csv')
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    dtype = torch.float32
    esm_emb = esm_embedding(ec_id_dict, device, dtype)
    esm_dist = get_dist_map(ec_id_dict, esm_emb, device, dtype)
    pickle.dump(esm_dist, open('./data/distance_map/' + train_file + '.pkl', 'wb'))
    pickle.dump(esm_emb, open('./data/distance_map/' + train_file + '_esm.pkl', 'wb'))
    
def prepare_infer_fasta(fasta_name):
    retrive_esm1b_embedding(fasta_name)
    csvfile = open('./data/' + fasta_name +'.csv', 'w', newline='')
    csvwriter = csv.writer(csvfile, delimiter = '\t')
    csvwriter.writerow(['Entry', 'EC number', 'Sequence'])
    fastafile = open('./data/' + fasta_name +'.fasta', 'r')
    for i in fastafile.readlines():
        if i[0] == '>':
            csvwriter.writerow([i.strip()[1:], ' ', ' '])
    
def mutate(seq: str, position: int) -> str:
    # 用于在给定的蛋白质序列 seq 中的特定位置 position 插入一个突变（这里用星号 '*' 表示）。
    # 分别获取突变位置之前和之后的序列部分。
    seql = seq[ : position]
    seqr = seq[position+1 : ]
    
    # 然后，这两部分序列通过插入星号 '*' 来合并，模拟在该位置的突变。
    # 最后，返回突变后的序列。
    seq = seql + '*' + seqr
    return seq

def mask_sequences(single_id, csv_name, fasta_name) :
    # 处理一个CSV文件，对指定的蛋白质序列（由 single_id 集合中的ID指定）进行多次（默认10次）的随机突变。
    
    csv_file = open('./data/'+ csv_name + '.csv')
    csvreader = csv.reader(csv_file, delimiter = '\t')
    output_fasta = open('./data/' + fasta_name + '.fasta','w')
    single_id = set(single_id)
    
    for i, rows in enumerate(csvreader):
        if rows[0] in single_id:
            for j in range(10):
                seq = rows[2].strip() # 蛋白质序列
                
                # 每次突变的比例是根据正态分布随机生成的，突变的次数是基于蛋白质序列长度和突变比例计算得出的。
                mu, sigma = .10, .02 # mean and standard deviation  mu = 0.10 表示平均突变比例为10%，标准差 sigma = 0.02 表示突变比例在均值周围的变化范围。
                s = np.random.normal(mu, sigma, 1) # 突变比例
                mut_rate = s[0] # 突变比例
                times = math.ceil(len(seq) * mut_rate) # 突变次数
                
                for k in range(times):
                    position = random.randint(1 , len(seq) - 1)
                    seq = mutate(seq, position)
                
                # 突变完成后，所有星号 '*' 被替换为 '<mask>'，模拟掩码或未知的氨基酸。
                seq = seq.replace('*', '<mask>')
                output_fasta.write('>' + rows[0] + '_' + str(j) + '\n')
                output_fasta.write(seq + '\n')

def mutate_single_seq_ECs(train_file):
    # 从一个CSV文件中获取 id_ec 和 ec_id 映射关系。
    id_ec, ec_id =  get_ec_id_dict('./data/' + train_file + '.csv')
    
    # 找出所有只有一个序列的EC编号（单序列EC编号），并检查这些EC编号关联的序列ID是否已经进行过突变处理。
    single_ec = set()
    for ec in ec_id.keys():
        if len(ec_id[ec]) == 1:
            single_ec.add(ec)
            
    single_id = set()
    for id in id_ec.keys():
        for ec in id_ec[id]:
            if ec in single_ec and not os.path.exists('./data/esm_data/' + id + '_1.pt'): # 如果该文件不存在,则该序列尚未突变
                single_id.add(id) # 只有那些既是单序列EC编号又没有对应突变序列文件的序列ID会被添加到 single_id 集合中，以便后续进行突变处理。
                break
    print("Number of EC numbers with only one sequences:",len(single_ec))
    print("Number of single-seq EC number sequences need to mutate: ",len(single_id))
    print("Number of single-seq EC numbers already mutated: ", len(single_ec) - len(single_id))
    
    mask_sequences(single_id, train_file, train_file+'_single_seq_ECs')
    fasta_name = train_file+'_single_seq_ECs'
    return fasta_name


# GitHub代码中，提到进行突变（mutation）的原因是为了处理那些只有一个序列的EC号，这些被称为"orphan" EC号。
# 在CLEAN模型的训练过程中，需要采样与锚点（anchor）序列不同的正例（positive）序列。
# 为了在这种情况下生成正例序列，他们对锚点序列进行了突变，以此作为正例序列。
# 这种做法是因为对比学习需要同时考虑正例和负例，以便学习不同序列之间的功能差异。

# 具体来说，对于那些只有一个序列的EC号，如果不进行突变，就无法为这些EC号提供多样化的正例序列进行训练。
# 通过突变产生的序列在功能上与原始序列相似，但在序列上有所不同，这样就能为模型提供额外的训练数据，从而提高模型对于这些稀有EC号的预测能力和准确性。

# 这个突变步骤是为了丰富那些只有单个序列代表的EC号的训练数据，从而使模型能够更好地学习和预测这些稀有但重要的酶功能。