import pandas as pd # 导入pandas 模块
# 读取数据
data_csv = pd.read_csv(r"./vfdb.csv") # 读取刚才写入的文件
data_csv1 = pd.read_csv(r"./VFDB_seq_code.csv") # 读取刚才写入的文件
add_column = data_csv1["VF CID"]


data_csv['label'] = add_column # 新增列sresult 并写入数据
data_csv.to_csv("./vfdb_process.csv", index=False) # 将新增的列数据，增加到原始数据中

