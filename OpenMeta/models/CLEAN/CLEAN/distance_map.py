import torch
from tqdm import tqdm
import torch.nn.functional as F

def get_cluster_center(model_emb, ec_id_dict):
    # 这个函数用于计算每个EC（Enzyme Commission）编号群组的嵌入表示的聚类中心。
    # ec_id_dict 是一个字典，包含EC编号到对应序列ID的映射。
    # 函数遍历每个EC编号，对该编号下的所有序列嵌入表示求平均值，得到该EC编号群组的聚类中心。

    cluster_center_model = {}
    id_counter = 0
    with torch.no_grad():
        for ec in tqdm(list(ec_id_dict.keys())):
            ids_for_query = list(ec_id_dict[ec])
            id_counter_prime = id_counter + len(ids_for_query) # 它代表了处理完当前EC编号后的新索引位置。
            emb_cluster = model_emb[id_counter: id_counter_prime]
            cluster_center = emb_cluster.mean(dim=0)
            cluster_center_model[ec] = cluster_center.detach().cpu()
            id_counter = id_counter_prime
    return cluster_center_model


def dist_map_helper_dot(keys1, lookup1, keys2, lookup2):
    # 这两个函数用于计算两个EC编号群组之间的距离。
    # 点积距离
    
    dist = {}
    lookup1 = F.normalize(lookup1, dim=-1, p=2)
    lookup2 = F.normalize(lookup2, dim=-1, p=2)
    for i, key1 in tqdm(enumerate(keys1)):
        current = lookup1[i].unsqueeze(0)
        dist_norm = (current - lookup2).norm(dim=1, p=2)
        dist_norm = dist_norm**2
        #dist_norm = (current - lookup2).norm(dim=1, p=2)
        dist_norm = dist_norm.detach().cpu().numpy()
        dist[key1] = {}
        for j, key2 in enumerate(keys2):
            dist[key1][key2] = dist_norm[j]
    return dist


def dist_map_helper(keys1, lookup1, keys2, lookup2):
    # 欧式距离
    
    dist = {}
    for i, key1 in tqdm(enumerate(keys1)):
        current = lookup1[i].unsqueeze(0)
        dist_norm = (current - lookup2).norm(dim=1, p=2)
        dist_norm = dist_norm.detach().cpu().numpy()
        dist[key1] = {}
        for j, key2 in enumerate(keys2):
            dist[key1][key2] = dist_norm[j]
    return dist


def get_dist_map(ec_id_dict, esm_emb, device, dtype, model=None, dot=False):
    '''
    Get the distance map for training, size of (N_EC_train, N_EC_train)
    between all possible pairs of EC cluster centers
    '''
    # inference all queries at once to get model embedding
    if model is not None:
        model_emb = model(esm_emb.to(device=device, dtype=dtype))
    else:
        # the first distance map before training comes from ESM
        model_emb = esm_emb
    # calculate cluster center by averaging all embeddings in one EC
    cluster_center_model = get_cluster_center(model_emb, ec_id_dict)  # 获取每个EC编号的聚类中心。
    
    # organize cluster centers in a matrix
    # 矩阵（model_lookup），用于存储所有EC编号群组的嵌入表示的聚类中心。
    total_ec_n, out_dim = len(ec_id_dict.keys()), model_emb.size(1)
    model_lookup = torch.zeros(total_ec_n, out_dim, device=device, dtype=dtype)
    
    # 这行代码获取了之前计算出的每个EC编号聚类中心的键（即EC编号）列表。
    ecs = list(cluster_center_model.keys())
    
    for i, ec in enumerate(ecs):
        model_lookup[i] = cluster_center_model[ec] # 对于每个EC编号，model_lookup[i] = cluster_center_model[ec] 将对应的聚类中心嵌入表示赋值到 model_lookup 矩阵的第 i 行。
    model_lookup = model_lookup.to(device=device, dtype=dtype) # model_lookup 的每一行 代表 EC编号
    
    # calculate pairwise distance map between total_ec_n * total_ec_n pairs
    print(f'Calculating distance map, number of unique EC is {total_ec_n}')
    if dot:
        model_dist = dist_map_helper_dot(ecs, model_lookup, ecs, model_lookup)
    else:
        model_dist = dist_map_helper(ecs, model_lookup, ecs, model_lookup)
    
    return model_dist # 包含所有EC编号对之间距离的映射。


def get_dist_map_test(model_emb_train, model_emb_test,
                      ec_id_dict_train, id_ec_test,
                      device, dtype, dot=False):
    '''
    Get the pair-wise distance map for test queries and train EC cluster centers
    map is of size of (N_test_ids, N_EC_train)
    '''
    print("The embedding sizes for train and test:",
          model_emb_train.size(), model_emb_test.size())
    # get cluster center for all EC appeared in training set
    cluster_center_model = get_cluster_center(
        model_emb_train, ec_id_dict_train)
    total_ec_n, out_dim = len(ec_id_dict_train.keys()), model_emb_train.size(1)
    model_lookup = torch.zeros(total_ec_n, out_dim, device=device, dtype=dtype)
    ecs = list(cluster_center_model.keys())
    for i, ec in enumerate(ecs):
        model_lookup[i] = cluster_center_model[ec]
    model_lookup = model_lookup.to(device=device, dtype=dtype)
    # calculate distance map between n_query_test * total_ec_n (training) pairs
    ids = list(id_ec_test.keys())
    print(f'Calculating eval distance map, between {len(ids)} test ids '
          f'and {total_ec_n} train EC cluster centers')
    if dot:
        eval_dist = dist_map_helper_dot(ids, model_emb_test, ecs, model_lookup)
    else:
        eval_dist = dist_map_helper(ids, model_emb_test, ecs, model_lookup)
    return eval_dist


def get_random_nk_dist_map(emb_train, rand_nk_emb_train,
                           ec_id_dict_train, rand_nk_ids,
                           device, dtype, dot=False):
    '''
    Get the pair-wise distance map between 
    randomly chosen nk ids from training and all EC cluster centers 
    map is of size of (nk, N_EC_train)
    '''
    cluster_center_model = get_cluster_center(emb_train, ec_id_dict_train)
    total_ec_n, out_dim = len(ec_id_dict_train.keys()), emb_train.size(1)
    model_lookup = torch.zeros(total_ec_n, out_dim, device=device, dtype=dtype)
    ecs = list(cluster_center_model.keys())
    for i, ec in enumerate(ecs):
        model_lookup[i] = cluster_center_model[ec]
    model_lookup = model_lookup.to(device=device, dtype=dtype)
    if dot:
        random_nk_dist_map = dist_map_helper_dot(
            rand_nk_ids, rand_nk_emb_train, ecs, model_lookup)
    else:
        random_nk_dist_map = dist_map_helper(
            rand_nk_ids, rand_nk_emb_train, ecs, model_lookup)
    return random_nk_dist_map
