import networkx as nx
import pickle
import os
import sys
from utils import create_dir

city = sys.argv[1]
output_path = sys.argv[2]
path = '../data/' + city + '/'
data_path = path + output_path + '/'

def process_client(client_id, client_data_path):
    """
    为每个客户端构建网格和道路的映射关系。
    """
    print(f"Processing client {client_id}...")
    client_pkl_path = os.path.join(client_data_path, 'used_pkl/')
    create_dir(client_pkl_path)

    trace_graph = nx.read_gml(os.path.join(client_data_path, 'trace_graph.gml'), destringizer=int)

    # 构建 grid2traceid_dict
    grid2traceid_dict = {}
    for k, v in dict(trace_graph.nodes()).items():
        pair = (v['gridx'], v['gridy'])
        if pair in grid2traceid_dict.keys():
            print(f"Warning: Duplicate grid found for client {client_id} at {pair}!")
        grid2traceid_dict[pair] = k

    pickle.dump(grid2traceid_dict, open(os.path.join(client_pkl_path, 'grid2traceid_dict.pkl'), 'wb'))

# 遍历所有客户端文件夹
for folder_name in os.listdir(data_path):
    if folder_name.startswith('client'):
        client_id = folder_name.split('_')[1]  # 提取客户端 ID
        client_data_path = os.path.join(data_path, folder_name)
        process_client(client_id, client_data_path)