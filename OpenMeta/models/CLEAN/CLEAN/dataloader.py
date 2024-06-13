import torch
import random
from .utils import format_esm, extract_rows


def mine_hard_negative(dist_map, knn=10):
    # 在 mine_hard_negative 函数中，这段代码的目的是从排序好的距离映射 
    
    print("The number of unique EC numbers: ", len(dist_map.keys()))
    ecs = list(dist_map.keys())
    negative = {}
    for i, target in enumerate(ecs):
        sort_orders = sorted(
            dist_map[target].items(), key=lambda x: x[1], reverse=False) # 与当前目标EC编号的其他EC编号及其相应的距离。这个列表是按照距离从小到大排序的
        if sort_orders[1][1] != 0: 
            # if sort_orders[1][1] != 0：这个条件检查第二近的EC编号（sort_orders[1]，因为sort_orders[0]是目标自身）的距离是否不为零。如果不为零，说明它是一个有效的负样本。
            freq = [1/i[1] for i in sort_orders[1:1 + knn]] # 这行代码计算接下来的 knn 个EC编号的倒数距离，用作权重。倒数距离是为了将较近的样本赋予更大的权重。
            neg_ecs = [i[0] for i in sort_orders[1:1 + knn]] # 这行代码收集这些EC编号。
        elif sort_orders[2][1] != 0:
            freq = [1/i[1] for i in sort_orders[2:2+knn]]
            neg_ecs = [i[0] for i in sort_orders[2:2+knn]]
        elif sort_orders[3][1] != 0:
            freq = [1/i[1] for i in sort_orders[3:3+knn]]
            neg_ecs = [i[0] for i in sort_orders[3:3+knn]]
        else:
            freq = [1/i[1] for i in sort_orders[4:4+knn]]
            neg_ecs = [i[0] for i in sort_orders[4:4+knn]]

        normalized_freq = [i/sum(freq) for i in freq]
        negative[target] = {
            'weights': normalized_freq,
            'negative': neg_ecs
        }
    return negative


def mine_negative(anchor, id_ec, ec_id, mine_neg):
    anchor_ec = id_ec[anchor]
    pos_ec = random.choice(anchor_ec)  # 这一行选择了一个与锚点样本 (anchor) 相关的EC编号作为正样本。anchor_ec 是锚点样本所关联的所有EC编号的列表，
    neg_ec = mine_neg[pos_ec]['negative']
    weights = mine_neg[pos_ec]['weights'] # 权重的设定是为了在选择负样本时引入概率因素。权重基于距离的倒数计算，意味着距离越近的负样本被选中的概率越大。
    
    result_ec = random.choices(neg_ec, weights=weights, k=1)[0]  # weights 指定了选择每个样本的概率权重。k=1 表示只选择一个元素。
    
    # 为了确保选取的负样本ID (result_ec) 不在锚点样本的EC编号列表 (anchor_ec) 中。
    while result_ec in anchor_ec:
        result_ec = random.choices(neg_ec, weights=weights, k=1)[0]
    neg_id = random.choice(ec_id[result_ec])
    return neg_id

def random_positive(id, id_ec, ec_id):
    pos_ec = random.choice(id_ec[id])
    pos = id
    if len(ec_id[pos_ec]) == 1:
        return pos + '_' + str(random.randint(0, 9))
    while pos == id:
        pos = random.choice(ec_id[pos_ec])
    return pos

def generate_triplets(batch, id_ec, ec_id, mine_neg, csv_path):
    prot_ids = batch['prot_ids']  # [272, 30]
    batch_size, seq_length = prot_ids.size()
    anchors = torch.empty(batch_size, seq_length, 1280)
    positives = torch.empty(batch_size, seq_length, 1280)
    negatives = torch.empty(batch_size, seq_length, 1280)
    full_list = []
    for ec in ec_id.keys():
        if '-' not in ec:
            full_list.append(ec)
    
    selected_rows = extract_rows(csv_path, prot_ids) # selected_rows[0][0] ['P52494', '3.2.1.28', 'MFTKNHRRMSSTSSDDDPFD...MHPEQRKQYK']    
    for i in range(batch_size):
        for j in range(seq_length):
            csv_path = csv_path
            selected_rows = extract_rows(csv_path, prot_ids) # selected_rows[0][0] ['P52494', '3.2.1.28', 'MFTKNHRRMSSTSSDDDPFD...MHPEQRKQYK']       
            if selected_rows[i][j] == None:
                continue
            anchor_ecs  = selected_rows[i][j][1].split(';')  
            anchor_ec = random.choice(anchor_ecs) # ['2.7.7.3', '3.6.1.73'],选一个
            anchor = random.choice(ec_id[anchor_ec])
            pos = random_positive(anchor, id_ec, ec_id) # 返回的是id，即name
            neg = mine_negative(anchor, id_ec, ec_id, mine_neg)            

            anchors[i, j] = format_esm(torch.load('./data/esm_data_171/' + anchor + '.pt'))
            positives[i, j] = format_esm(torch.load('./data/esm_data_171/' + pos + '.pt'))
            negatives[i, j] = format_esm(torch.load('./data/esm_data_171/' + neg + '.pt'))

    return anchors, positives, negatives # all: torch.Size([272, 30, 1280])


class Triplet_dataset_with_mine_EC_glm(torch.utils.data.Dataset):
    def __init__(self, id_ec, ec_id, mine_neg, prot_ids):
        self.prot_ids = prot_ids  # 直接使用 prot_ids 列表
        self.id_ec = id_ec
        self.ec_id = ec_id
        self.mine_neg = mine_neg
        self.full_list = []
        self.mine_neg = mine_neg
        for ec in ec_id.keys():
            if '-' not in ec:
                self.full_list.append(ec)

    def __len__(self):
        return self.prot_ids.size(0)

    def __getitem__(self, index):  
        # anchor_ec = self.full_list[index]
        # anchor = random.choice(self.ec_id[anchor_ec])
        
        anchor_ecs_ids = self.prot_ids[index] #len=30,[272,30], anchor_ecs_ids tensor([6751, 6752, 6753, 6754, 6755, 6756, 6757, 6758, 6759, 6760, 6761, 6762, ..., 6780])
        anchor_ecs = self.full_list[index] # '2.3.1.48',len=3082
        anchors = torch.empty(self.prot_ids.size(1),1280)
        
        for j in range(self.prot_ids.size(1)):
            anchor_ecs_id = anchor_ecs_ids[j].item() # 6751
            anchor_ec = self.full_list[anchor_ecs_id] 
            anchor_ec_str = str(anchor_ec.item())
            anchor = random.choice(self.ec_id[anchor_ec_str])
            anchor_data = torch.load('./data/esm_data_171/' + anchor + '.pt')
            anchors[j] = format_esm(anchor_data)

        pos_ids = []
        neg_ids = []
        positives = torch.empty(self.prot_ids.size(1), 1280)
        negatives = torch.empty(self.prot_ids.size(1), 1280)
        for j in range(self.prot_ids.size(1)):
            anchor_ec = anchor_ecs[j]
            anchor_ec_str = str(anchor_ec.item())
            anchor = random.choice(self.ec_id[anchor_ec_str])
            pos = random_positive(anchor, self.id_ec, self.ec_id)
            neg = mine_negative(anchor, self.id_ec, self.ec_id, self.mine_neg)

            anchor_data = torch.load('./data/esm_data_171/' + anchor + '.pt')
            pos_data = torch.load('./data/esm_data_171/' + pos + '.pt')
            neg_data = torch.load('./data/esm_data_171/' + neg + '.pt')

            anchors[j] = format_esm(anchor_data)
            positives[j] = format_esm(pos_data)
            negatives[j] = format_esm(neg_data)
            pos_ids.append(pos)
            neg_ids.append(neg)

        return anchors, positives, negatives, pos_ids, neg_ids


class Triplet_dataset_with_mine_EC2(torch.utils.data.Dataset):
    def __init__(self, id_ec, ec_id, mine_neg, prot_ids):
        self.id_ec = id_ec
        self.ec_id = ec_id
        self.full_list = [] #len=3082 ['2.5.1.18', '3.2.1.4', '2.7.11.1', '1.14.13.84', '5.6.2.2', '1.8.7.3', '3.1.21.10', '6.3.2.39', '2.3.1.32', '1.1.1.37', '7.2.1.1', '5.3.3.12', '2.3.2.27', '7.1.1.2', ...]
        self.mine_neg = mine_neg
        for ec in ec_id.keys():
            if '-' not in ec:
                self.full_list.append(ec)
        self.prot_ids = prot_ids
    
    def __len__(self):
        return len(self.full_list)
    
    def __getitem__(self, index):
        # 处理整个batch而不是单个样本
        batch_size, seq_length = self.prot_ids.size()
        anchors = torch.empty(batch_size, seq_length, 1280)
        positives = torch.empty(batch_size, seq_length, 1280)
        negatives = torch.empty(batch_size, seq_length, 1280)

        for i in range(batch_size):
            for j in range(seq_length):
                # 生成anchor, positive, negative
                anchor_ec = self.full_list[index] # '5.3.3.7'
                anchor = random.choice(self.ec_id[anchor_ec]) 
                pos = random_positive(anchor, self.id_ec, self.ec_id)
                neg = mine_negative(anchor, self.id_ec, self.ec_id, self.mine_neg)

                # 加载并格式化embeddings
                anchors[i, j] = format_esm(torch.load('./data/esm_data_171/' + anchor + '.pt'))
                positives[i, j] = format_esm(torch.load('./data/esm_data_171/' + pos + '.pt'))
                negatives[i, j] = format_esm(torch.load('./data/esm_data_171/' + neg + '.pt'))

        return anchors, positives, negatives

class Triplet_dataset_with_mine_EC(torch.utils.data.Dataset):    
    # 定义一个数据集类，用于生成三元组（锚点、正样本、负样本）。
    def __init__(self, id_ec, ec_id, mine_neg, prot_ids):
        self.id_ec = id_ec
        self.ec_id = ec_id
        self.full_list = [] #len=3082 ['2.5.1.18', '3.2.1.4', '2.7.11.1', '1.14.13.84', '5.6.2.2', '1.8.7.3', '3.1.21.10', '6.3.2.39', '2.3.1.32', '1.1.1.37', '7.2.1.1', '5.3.3.12', '2.3.2.27', '7.1.1.2', ...]
        self.mine_neg = mine_neg
        for ec in ec_id.keys():
            if '-' not in ec:
                self.full_list.append(ec)
        self.prot_ids = prot_ids

    def __len__(self):
        return len(self.full_list)
    def __getitem__(self, index):
        anchor_ec = self.full_list[index] # '5.3.3.7'
        anchor = random.choice(self.ec_id[anchor_ec]) 
        pos = random_positive(anchor, self.id_ec, self.ec_id)
        neg = mine_negative(anchor, self.id_ec, self.ec_id, self.mine_neg)
        # a = torch.load('./data/esm_data/' + anchor + '.pt')
        # p = torch.load('./data/esm_data/' + pos + '.pt')
        # n = torch.load('./data/esm_data/' + neg + '.pt')
        
        a = torch.load('./data/esm_data_171/' + anchor + '.pt')
        p = torch.load('./data/esm_data_171/' + pos + '.pt')
        n = torch.load('./data/esm_data_171/' + neg + '.pt')
        
        return format_esm(a), format_esm(p), format_esm(n)


class MultiPosNeg_dataset_with_mine_EC(torch.utils.data.Dataset):
    
    # 定义一个数据集类，用于生成包含多个正样本和负样本的样本集。
    # 根据索引生成包含多个正负样本的数据。

    def __init__(self, id_ec, ec_id, mine_neg, n_pos, n_neg):
        self.id_ec = id_ec
        self.ec_id = ec_id
        self.n_pos = n_pos
        self.n_neg = n_neg
        self.full_list = []
        self.mine_neg = mine_neg
        for ec in ec_id.keys():
            if '-' not in ec:
                self.full_list.append(ec)

    def __len__(self):
        return len(self.full_list)

    def __getitem__(self, index):
        anchor_ec = self.full_list[index]
        anchor = random.choice(self.ec_id[anchor_ec])
        a = format_esm(torch.load('./data/esm_data/' +
                       anchor + '.pt')).unsqueeze(0)
        data = [a]
        for _ in range(self.n_pos):
            pos = random_positive(anchor, self.id_ec, self.ec_id)
            p = format_esm(torch.load('./data/esm_data/' +
                           pos + '.pt')).unsqueeze(0)
            data.append(p)
        for _ in range(self.n_neg):
            neg = mine_negative(anchor, self.id_ec, self.ec_id, self.mine_neg)
            n = format_esm(torch.load('./data/esm_data/' +
                           neg + '.pt')).unsqueeze(0)
            data.append(n)
        return torch.cat(data)
