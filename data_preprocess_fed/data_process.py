import os
import os
import pickle
import random
import json
import math
import sys
import heapq
import re

from tqdm import tqdm

from utils import create_dir, get_border

city = sys.argv[1]
num_clients = int(sys.argv[2])  # 新增的参数，client 数量
sample_rates_str = sys.argv[3]  # 包含采样率的字符串
input_path = sys.argv[4]
output_path = sys.argv[5]

# 将字符串转换为浮点数列表
client_sample_rates = list(map(float, sample_rates_str.split()))

MIN_LAT, MIN_LNG, MAX_LAT, MAX_LNG = get_border('../data/' + city + '_road.txt')

def randomDownSampleBySize(sampleData: list, sampleRate: float) -> (list, list, list):
    """
        randomly sampling
    """
    resData, pureData, resIdx = [], [], []
    for i in range(len(sampleData)):
        trajList = sampleData[i]
        tempRes = [trajList[1]]  # 首节点
        tmpIdx = [0]
        for j in range(2, len(trajList) - 2):
            if random.random() <= sampleRate:
                tempRes.append(trajList[j])
                tmpIdx.append(j - 1)
        # tempRes.append(trajList[-2])  # 尾节点
        # tmpIdx.append(len(trajList) - 2)
        resData.append(tempRes)
        pureData.append(trajList)
        resIdx.append(tmpIdx)
    return resData, pureData, resIdx


class DataProcess:
    def __init__(self, traj_input_path, output_dir, num_clients, client_sample_rates, max_road_len=25, min_road_len=15):
        self.traj_input_path = traj_input_path
        self.output_dir = output_dir
        self.num_clients = num_clients
        self.client_sample_rates = client_sample_rates
        self.max_road_len = max_road_len
        self.min_road_len = min_road_len
        self.split_data_for_clients()

    def read_data_from_folder(self, folder_path):
        """
        Read data from a folder. The folder contains a 'trajectories.txt' file.
        """
        file_path = os.path.join(folder_path, 'trajectories.txt')
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file {file_path} does not exist.")
        with open(file_path, 'r') as file:
            data = file.readlines()
        return self.cutData(data)

    def cutData(self, traj_list):
        """
        Ensure each trace's length is in [min_road_len+1, max_road_len+min_road_len+1)
        """
        finalLs = []
        tempLs = list()  # 用来保存单个轨迹
        for idx, sen in enumerate(traj_list):
            if sen[0] == '#':  # 表明是一条轨迹的开头
                if idx != 0:
                    finalLs.append(tempLs)
                tempLs = [sen]
            else:  # 增加轨迹点
                tempLs.append(sen)
        finalLs.append(tempLs)

        cutFinalLs = []
        for traces in finalLs:
            assert traces[0][0] == '#'
            title = traces[0]
            traces = traces[1:]
            lens = len(traces)
            if lens < self.min_road_len:
                continue
            if lens < self.max_road_len and lens >= self.min_road_len:
                cutFinalLs.append([title] + traces)
            else:
                cutnum = lens / self.max_road_len
                int_cutnum = int(cutnum)
                lstnum = lens - int_cutnum * self.max_road_len
                if lens % self.max_road_len != 0:
                    int_cutnum += 1
                else:
                    lstnum = self.max_road_len
                if lstnum < self.min_road_len:
                    int_cutnum -= 1

                for i in range(int_cutnum - 1):
                    tmp_ls = [title] + traces[i * self.max_road_len:(i + 1) * self.max_road_len]
                    cutFinalLs.append(tmp_ls)

                latLS = [title] + traces[(int_cutnum - 1) * self.max_road_len:]
                cutFinalLs.append(latLS)

        for i in cutFinalLs:
            assert len(i) >= 16 and len(i) <= 40
        return cutFinalLs


    # def sampling(self, sample_data, sample_rate):
    #     """
    #     Down sampling
    #     """
    #     path = '../data/' + city + '/real_distances.pkl'
    #     downsampleData, pureData, downsampleIdx = randomDownSampleBySize(sample_data, sample_rate)
    #     traces_ls, tgt_roads_ls, candidates_id_ls, time_stamp_ls = [], [], [], []
    #     with open(path, 'rb') as file:
    #         road_distance_data = pickle.load(file)
    #         for downdata in tqdm(downsampleData):
    #             traces, roads, candidates_ids, time_stamp = [], [], [], []
    #             for i in downdata:
    #                 if i[0] == '#' or i[0] == '\n':
    #                     continue
    #                 il = i.split(',')
    #                 time_stamp.append(il[0])
    #                 lat, lng = float(il[1]), float(il[2])
    #                 traces.append((lat, lng))
    #                 candidates_id = self.get_road_candidates(int(i.split(',')[3]), road_distance_data)
    #                 candidates_ids.append(candidates_id)
    #                 roads.append(candidates_id.index(int(i.split(',')[3])))
    #
    #             traces_ls.append(traces)
    #             tgt_roads_ls.append(roads)
    #             candidates_id_ls.append(candidates_ids)
    #             time_stamp_ls.append(time_stamp)
    #     return traces_ls, tgt_roads_ls, candidates_id_ls, time_stamp_ls, downsampleIdx, downsampleData

    def sampling(self, sample_data, sample_rate):
        """
            down sampling
        """
        downsampleData, pureData, downsampleIdx = randomDownSampleBySize(sample_data, sample_rate)
        traces_ls, roads_ls = [], []
        for downdata, puredata in zip(downsampleData, pureData):
            traces, roads = [], []
            for i in downdata:
                if i[0] == '#' or i[0] == '\n':
                    continue
                il = i.split(',')
                lat, lng = float(il[1]), float(il[2])
                traces.append((lat, lng))
            for i in puredata:
                if i[0] == '#' or i[0] == '\n':
                    continue
                roads.append(int(i.split(',')[3]))
            traces_ls.append(traces)
            roads_ls.append(roads)
        return traces_ls, roads_ls, downsampleIdx, downsampleData


    def splitData(self, output_dir, traces_ls, roads_ls, downsampleIdx, downsampleData, train_rate=0.7, val_rate=0.2):
        """
            split original data to train, valid and test datasets
        """
        create_dir(output_dir)
        create_dir(output_dir + 'data_split/')
        train_data_dir = output_dir + 'train_data/'
        create_dir(train_data_dir)
        val_data_dir = output_dir + 'val_data/'
        create_dir(val_data_dir)
        test_data_dir = output_dir + 'test_data/'
        create_dir(test_data_dir)
        num_sample = len(traces_ls)
        train_size, val_size = int(num_sample * train_rate), int(num_sample * val_rate)
        idxs = list(range(num_sample))
        random.shuffle(idxs)
        train_idxs = idxs[:train_size]
        val_idxs = idxs[train_size:train_size + val_size]
        trainset, valset, testset = [], [], []

        train_trace = []
        val_trace = []
        test_trace = []
        for i in range(num_sample):
            if i in train_idxs:
                trainset.extend([traces_ls[i], roads_ls[i], downsampleIdx[i]])
                train_trace += [downsampleData[i]]
            elif i in val_idxs:
                valset.extend([traces_ls[i], roads_ls[i], downsampleIdx[i]])
                val_trace += [downsampleData[i]]
            else:
                testset.extend([traces_ls[i], roads_ls[i], downsampleIdx[i]])
                test_trace += [downsampleData[i]]

        with open(os.path.join(train_data_dir, "train.json"), 'w') as fp:
            json.dump(trainset, fp)

        with open(os.path.join(val_data_dir, "val.json"), 'w') as fp:
            json.dump(valset, fp)

        with open(os.path.join(test_data_dir, "test.json"), 'w') as fp:
            json.dump(testset, fp)

        all_trace = [train_trace, val_trace, test_trace]
        all_trace_name = ['train_trace.txt', 'val_trace.txt', 'test_trace.txt']
        for i in range(3):
            tmptrace = all_trace[i]
            path = output_dir + 'data_split/' + all_trace_name[i]
            with open(path, 'w') as f:
                for traces in tmptrace:
                    for trace in traces:
                        f.write(trace)

        with open(os.path.join(output_dir, 'data_split', 'downsample_trace.txt'), 'w') as f:
            for traces in downsampleData:
                for trace in traces:
                    f.write(trace)

    def split_data_for_clients(self):
        """
        Split data for each client and apply client-specific sampling rates.
        """
        for client_id in range(self.num_clients):
            folder_path = os.path.join(self.traj_input_path, f'dataset_{client_id + 1}')
            client_data = self.read_data_from_folder(folder_path)
            sample_rate = self.client_sample_rates[client_id]
            client_output_dir = os.path.join(self.output_dir, f'client_{client_id}/')
            create_dir(client_output_dir)
            traces_ls, roads_ls, downsampleIdx, downsampleData = self.sampling(
                client_data, sample_rate)
            self.splitData(client_output_dir, traces_ls, roads_ls, downsampleIdx, downsampleData)




if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python data_process.py <city> <num_clients> <sample_rates>")
        sys.exit(1)

    city = sys.argv[1]
    num_clients = int(sys.argv[2])  # 新增的参数，client 数量
    sample_rates_str = sys.argv[3]  # 包含采样率的字符串

    # 将字符串转换为浮点数列表
    client_sample_rates = list(map(float, sample_rates_str.split()))

    MIN_LAT, MIN_LNG, MAX_LAT, MAX_LNG = get_border('../data/' + city + '_road.txt')

    path = '../data/' + city
    input_path = path + '/' + input_path + '/'
    output_path = path + '/' + output_path + '/'

    data_processor = DataProcess(
        traj_input_path=input_path,
        output_dir=output_path,
        num_clients=num_clients,
        client_sample_rates=client_sample_rates
    )