import networkx as nx
import torch
import os
import sys
from utils import gps2grid, grid2gps, create_dir, get_border

city = sys.argv[1]
output_path = sys.argv[2]
MIN_LAT, MIN_LNG, MAX_LAT, MAX_LNG = get_border('../data/' + city + '_road.txt')

def get_data_from_client(client_folder_path):
    """
    从指定的客户端文件夹读取数据，构建 grid2id_dict 和 trace_dict。
    """
    grid2id_dict = {}
    trace_dict = {}
    file_path = os.path.join(client_folder_path, "data_split/downsample_trace.txt")
    with open(file_path, 'r') as f:
        trace_ls = f.readlines()

    lst_id = -1
    for trace in trace_ls:
        if trace.startswith('#') or trace.startswith("\n"):
            lst_id = -1
            continue
        lng = float(trace.split(',')[2])
        lat = float(trace.split(',')[1])

        gridx, gridy = gps2grid(lat, lng, MIN_LAT=MIN_LAT, MIN_LNG=MIN_LNG)
        if gridx < 0 or gridy < 0:
            print(lat, lng, gridx, gridy)
        if (gridx, gridy) not in grid2id_dict.keys():
            grid2id_dict[(gridx, gridy)] = len(grid2id_dict)
        tmp_id = grid2id_dict[(gridx, gridy)]
        if lst_id != -1:
            if lst_id == tmp_id:
                continue
            if (lst_id, tmp_id) not in trace_dict.keys():
                trace_dict[(lst_id, tmp_id)] = 1
            else:
                trace_dict[(lst_id, tmp_id)] += 1
        lst_id = tmp_id
    return grid2id_dict, trace_dict

def get_global_data(base_path):
    """
    从所有客户端读取数据，构建全局的 grid2id_dict 和 trace_dict。
    """
    grid2id_dict = {}
    trace_dict = {}

    # 遍历所有以 'client' 开头的子文件夹
    for folder_name in os.listdir(base_path):
        if folder_name.startswith('client'):
            folder_path = os.path.join(base_path, folder_name)
            # 获取子文件夹中的文件
            file_path = os.path.join(folder_path, "data_split/downsample_trace.txt")
            with open(file_path, 'r') as f:
                trace_ls = f.readlines()

            lst_id = -1
            for trace in trace_ls:
                if trace.startswith('#') or trace.startswith("\n"):
                    lst_id = -1
                    continue
                lng = float(trace.split(',')[2])
                lat = float(trace.split(',')[1])

                gridx, gridy = gps2grid(lat, lng, MIN_LAT=MIN_LAT, MIN_LNG=MIN_LNG)
                if gridx < 0 or gridy < 0:
                    print(lat, lng, gridx, gridy)
                if (gridx, gridy) not in grid2id_dict.keys():
                    grid2id_dict[(gridx, gridy)] = len(grid2id_dict)
                tmp_id = grid2id_dict[(gridx, gridy)]
                if lst_id != -1:
                    if lst_id == tmp_id:
                        continue
                    if (lst_id, tmp_id) not in trace_dict.keys():
                        trace_dict[(lst_id, tmp_id)] = 1
                    else:
                        trace_dict[(lst_id, tmp_id)] += 1
                lst_id = tmp_id
    return grid2id_dict, trace_dict

def build_graph(grid2id_dict, trace_dict):
    """
    构建轨迹图
    """
    G = nx.DiGraph()
    weighted_edges = []
    for k, v in trace_dict.items():
        weighted_edges.append((k[0], k[1], v))
    G.add_weighted_edges_from(weighted_edges)
    for k, v in grid2id_dict.items():
        if v not in G.nodes():
            G.add_node(v)
        G.nodes[v]['gridx'] = k[0]
        G.nodes[v]['gridy'] = k[1]
    return G

def build_pyG(G):
    """
    为轨迹图构建特征和 edge_index
    """
    edge_index = [[], []]
    x = []
    weight = []
    for i in G.edges():
        edge_index[0].append(i[0])
        edge_index[1].append(i[1])
        tmp_weight = G[i[0]][i[1]]['weight']
        weight.append(tmp_weight)

    inweight = weight.copy()
    outweight = weight.copy()

    for i in G.nodes:
        x.append(grid2gps(G.nodes[i]['gridx'], G.nodes[i]['gridy'], G.nodes[i]['gridx'], G.nodes[i]['gridy'],
                          MIN_LAT=MIN_LAT, MIN_LNG=MIN_LNG))

    out_edge_index = torch.tensor(edge_index)
    in_edge_index = out_edge_index[[1, 0]]
    x = torch.tensor(x)
    inweight = torch.tensor(inweight)
    outweight = torch.tensor(outweight)
    return x, in_edge_index, inweight, out_edge_index, outweight

def save_graph(G, x, in_edge_index, inweight, out_edge_index, outweight, save_path):
    """
    保存轨迹图和 PyG 格式的数据
    """
    create_dir(save_path)
    nx.write_gml(G, os.path.join(save_path, "trace_graph.gml"))
    trace_graph_pt_path = os.path.join(save_path, 'trace_graph_pt')
    create_dir(trace_graph_pt_path)
    torch.save(in_edge_index, os.path.join(trace_graph_pt_path, 'in_edge_index.pt'))
    torch.save(x, os.path.join(trace_graph_pt_path, 'x.pt'))
    torch.save(inweight, os.path.join(trace_graph_pt_path, 'inweight.pt'))
    torch.save(outweight, os.path.join(trace_graph_pt_path, 'outweight.pt'))
    torch.save(out_edge_index, os.path.join(trace_graph_pt_path, 'out_edge_index.pt'))

def build_graph_with_mapping(grid2id_dict, trace_dict, global_grid2id_dict):
    """
    构建轨迹图，并生成本地到全局节点的映射
    Args:
        grid2id_dict: 本地的 grid2id 映射
        trace_dict: 本地的轨迹字典
        global_grid2id_dict: 全局的 grid2id 映射
    Returns:
        G: 构建的本地图
        local_to_global: 本地节点到全局节点的映射
    """
    G = nx.DiGraph()
    weighted_edges = []
    local_to_global = {}

    for k, v in trace_dict.items():
        weighted_edges.append((k[0], k[1], v))
    G.add_weighted_edges_from(weighted_edges)

    for k, v in grid2id_dict.items():
        if v not in G.nodes():
            G.add_node(v)
        G.nodes[v]['gridx'] = k[0]
        G.nodes[v]['gridy'] = k[1]

        # 构建本地到全局的节点映射
        if k in global_grid2id_dict:
            local_to_global[v] = global_grid2id_dict[k]

    return G, local_to_global

def save_mapping(local_to_global, save_path):
    """
    保存本地到全局的节点映射
    """
    mapping_path = os.path.join(save_path, 'local_to_global_mapping.pt')
    torch.save(local_to_global, mapping_path)

if __name__ == "__main__":
    path = '../data/' + city + '/'
    data_path = path + output_path + '/'

    # 构建全局图
    grid2id_dict_global, trace_dict_global = get_global_data(data_path)
    G_global = build_graph(grid2id_dict_global, trace_dict_global)
    x_global, in_edge_index_global, inweight_global, out_edge_index_global, outweight_global = build_pyG(G_global)
    # 保存全局图
    save_graph(G_global, x_global, in_edge_index_global, inweight_global, out_edge_index_global, outweight_global, data_path)

    # 为每个客户端构建本地图
    for folder_name in os.listdir(data_path):
        if folder_name.startswith('client'):
            client_folder_path = os.path.join(data_path, folder_name)
            print(f"Processing {folder_name}...")

            grid2id_dict_client, trace_dict_client = get_data_from_client(client_folder_path)
            G_client, local_to_global = build_graph_with_mapping(grid2id_dict_client, trace_dict_client, grid2id_dict_global)
            x_client, in_edge_index_client, inweight_client, out_edge_index_client, outweight_client = build_pyG(G_client)

            # 保存客户端本地图
            save_graph(G_client, x_client, in_edge_index_client, inweight_client, out_edge_index_client, outweight_client, client_folder_path)

            # 保存本地到全局的节点映射
            save_mapping(local_to_global, client_folder_path)