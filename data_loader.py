import os.path as osp
import torch
import json
import pickle
from torch.utils.data import Dataset
import json
import os.path as osp
import pickle
from torch.utils.data import Dataset
import torch
import data_preprocess.utils as utils
import time
import numpy as np

class MyDataset(Dataset):
    def __init__(self, root_path, path, name):
        # parent_path = 'path'
        if not root_path.endswith('/'):
            root_path += '/'
        self.MIN_LAT, self.MIN_LNG, MAX_LAT, MAX_LNG = utils.get_border(root_path + 'road.txt')
        if not path.endswith('/'):
            path += '/'
        self.data_path = osp.join(path, f"{name}_data/{name}.json")
        self.map_path = osp.join(path, "used_pkl/grid2traceid_dict.pkl")
        self.buildingDataset(self.data_path)

    def buildingDataset(self, data_path, subset_ratio=0.8):
        grid2traceid_dict = pickle.load(open(self.map_path, 'rb'))
        self.traces_ls = []
        with open(data_path, "r") as fp:
            data = json.load(fp)
            total_data_points = len(data[0::3])
            subset_size = int(total_data_points * subset_ratio)

            # Adjust subset size to ensure it's within the range of available data
            subset_size = min(subset_size, total_data_points)

            # Select subset of data indices
            subset_indices = range(0, total_data_points, total_data_points // subset_size)[:subset_size]

            for idx in subset_indices:
                gps_ls = data[0::3][idx]
                traces = []
                for gps in gps_ls:
                    gridx, gridy = utils.gps2grid(gps[0], gps[1], MIN_LAT=self.MIN_LAT, MIN_LNG=self.MIN_LNG)
                    traces.append(grid2traceid_dict[(gridx, gridy)] + 1)
                self.traces_ls.append(traces)

            self.roads_ls = [data[1::3][idx] for idx in subset_indices]
            self.traces_gps_ls = [data[0::3][idx] for idx in subset_indices]
            self.sampleIdx_ls = [data[2::3][idx] for idx in subset_indices]

        self.length = len(self.traces_ls)
        assert len(self.traces_ls) == len(self.roads_ls)

    def __getitem__(self, index):
        return self.traces_ls[index], self.roads_ls[index],\
            self.traces_gps_ls[index], self.sampleIdx_ls[index]

    def __len__(self):
        return self.length


# def padding(batch):
#     trace_lens = [len(sample[0]) for sample in batch]
#     road_lens = [len(sample[1]) for sample in batch]
#     max_tlen, max_rlen = max(trace_lens), max(road_lens)
#     x, y, z, w = [], [], [], []
#     # 0: [PAD]
#     for sample in batch:
#         x.append(sample[0] + [0] * (max_tlen - len(sample[0])))
#         y.append(sample[1] + [-1] * (max_rlen - len(sample[1])))
#         z.append(sample[2] + [[0, 0]] * (max_tlen - len(sample[2])))
#         w.append(sample[3] + [-1] * (max_tlen - len(sample[3])))
#     f = torch.LongTensor
#     return f(x), f(y), torch.FloatTensor(z), f(w), trace_lens, road_lens



class FederatedDataset(Dataset):
    def __init__(self, data_path, path, name, city):
        if not path.endswith('/'):
            path += '/'
        self.data_path = osp.join(path, f"{name}_data/{name}.json")
        self.MIN_LAT, self.MIN_LNG, self.MAX_LAT, self.MAX_LNG = utils.get_border('data/' + city + '_road.txt')
        self.map_path = osp.join(path, "used_pkl/grid2traceid_dict.pkl")
        self.buildingDataset(self.data_path)
        if city == 'beijing':
            self.link_cnt = 8533
        else:
            self.link_cnt = 4254

    def buildingDataset(self, data_path, subset_ratio=1):
        grid2traceid_dict = pickle.load(open(self.map_path, 'rb'))
        self.traces_ls = []

        with open(data_path, "r") as fp:
            data = json.load(fp)
            total_data_points = len(data[0::3])
            subset_size = int(total_data_points * subset_ratio)

            # Adjust subset size to ensure it's within the range of available data
            subset_size = min(subset_size, total_data_points)

            # Select subset of data indices
            subset_indices = range(0, total_data_points, total_data_points // subset_size)[:subset_size]

            for idx in subset_indices:
                gps_ls = data[0::3][idx]
                traces = []
                for gps in gps_ls:
                    gridx, gridy = utils.gps2grid(gps[0], gps[1], MIN_LAT=self.MIN_LAT, MIN_LNG=self.MIN_LNG)
                    traces.append(grid2traceid_dict[(gridx, gridy)] + 1)
                self.traces_ls.append(traces)

            self.roads_ls = [data[1::3][idx] for idx in subset_indices]
            self.traces_gps_ls = [data[0::3][idx] for idx in subset_indices]
            self.sampleIdx_ls = [data[2::3][idx] for idx in subset_indices]

        self.length = len(self.traces_ls)

        self.cnt = self.count_point()

        print(self.data_path)
        print("Total points: ", self.cnt)
        # print(len(self.traces_ls), len(self.time_stamps), len(self.tgt_roads_ls), len(self.candidates_id))
        assert len(self.traces_ls) == len(self.roads_ls)

    def count_point(self):
        count = 0
        for trace in self.traces_ls:
            for point in trace:
                count += 1
        return count

    def __getitem__(self, index):
        return self.traces_ls[index], self.roads_ls[index], \
            self.traces_gps_ls[index], self.sampleIdx_ls[index]

    def __len__(self):
        return self.length

    # def padding(self, batch):
    #     trace_lens = [len(sample[0]) for sample in batch]
    #     candidates_lens = [len(candidates) for sample in batch for candidates in sample[3]]
    #     max_tlen, max_clen = max(trace_lens), max(candidates_lens)
    #     traces, time_stamp, tgt_roads, candidates_id = [], [], [], []
    #     for sample in batch:
    #         traces.append(sample[0] + [0] * (max_tlen - len(sample[0])))
    #         time_stamp.append(sample[1] + [-1] * (max_tlen - len(sample[1])))
    #         tgt_roads.append(sample[2] + [0] * (max_tlen - len(sample[2])))
    #         candidates_id.append(
    #             [candidates_id + [self.link_cnt] * (max_clen - len(candidates_id)) for candidates_id in sample[3]] + [
    #                 [self.link_cnt] * max_clen] * (max_tlen - len(sample[3])))
    #
    #     traces_array = np.array(traces)
    #     time_stamp_array = np.array(time_stamp)
    #     traces_tensor = torch.FloatTensor(traces_array).unsqueeze(-1)
    #     time_stamp_tensor = torch.FloatTensor(time_stamp_array).unsqueeze(-1)
    #     traces = torch.cat((traces_tensor, time_stamp_tensor), dim=-1)
    #     return traces, torch.LongTensor(tgt_roads), torch.LongTensor(candidates_id), trace_lens

def padding(batch):
    trace_lens = [len(sample[0]) for sample in batch]
    road_lens = [len(sample[1]) for sample in batch]
    max_tlen, max_rlen = max(trace_lens), max(road_lens)
    x, y, z, w = [], [], [], []
    # 0: [PAD]
    for sample in batch:
        x.append(sample[0] + [0] * (max_tlen - len(sample[0])))
        y.append(sample[1] + [-1] * (max_rlen - len(sample[1])))
        z.append(sample[2] + [[0, 0]] * (max_tlen - len(sample[2])))
        w.append(sample[3] + [-1] * (max_tlen - len(sample[3])))
    f = torch.LongTensor
    return f(x), f(y), torch.FloatTensor(z), f(w), trace_lens, road_lens


def load_federated_data(dataset, client_id, num_clients):
    data_len = len(dataset)
    client_data_len = data_len // num_clients
    client_start = client_id * client_data_len
    client_end = (client_id + 1) * client_data_len if client_id < num_clients - 1 else data_len
    subset = torch.utils.data.Subset(dataset, range(client_start, client_end))
    print(f"Client {client_id}: Subset length = {len(subset)}")
    return subset

if __name__ == "__main__":
    pass
