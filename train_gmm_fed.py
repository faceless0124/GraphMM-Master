import nni
import numpy as np
import torch
import torch.optim as optim
from config import get_params
from nni.utils import merge_parameter

from model.gmm import GMM
from data_loader import MyDataset, FederatedDataset, padding
from torch.utils.data import DataLoader
from graph_data import GraphData
from tqdm import tqdm
import torch.nn as nn
from sklearn.metrics import accuracy_score
import os.path as osp
from copy import deepcopy
from data_preprocess.utils import create_dir
import time
from collections import OrderedDict

from tqdm import tqdm
import torch.nn as nn


def train(model, train_iter, optimizer, device, gdata, args, client_id):
    model.train()
    train_l_sum, count = 0., 0

    with tqdm(total=len(train_iter), desc=f"Training (Client {client_id})", unit="batch") as pbar:
        for data in train_iter:
            grid_traces = data[0].to(device)
            tgt_roads = data[1].to(device)
            traces_gps = data[2].to(device)
            sample_Idx = data[3].to(device)
            traces_lens, road_lens = data[4], data[5]

            # 根据sample_idx处理tgt_road
            sample_Idx_masked = torch.where(sample_Idx == -1, 0, sample_Idx).cpu()
            result = tgt_roads[torch.arange(tgt_roads.size(0)).unsqueeze(1), sample_Idx_masked]
            result[sample_Idx == -1] = -1
            result = result.to(device)

            loss = model(grid_traces=grid_traces,
                         traces_gps=traces_gps,
                         traces_lens=traces_lens,
                         road_lens=traces_lens,
                         tgt_roads=result,
                         gdata=gdata,
                         sample_Idx=sample_Idx,
                         tf_ratio=args['tf_ratio'])

            train_l_sum += loss.item()
            count += 1

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.)
            optimizer.step()

            pbar.set_postfix(train_loss=loss.item())
            pbar.update(1)

    return train_l_sum / count


# def evaluate(model, eval_iter, device, gdata, use_crf):
#     model.eval()
#     eval_acc_sum, count = 0., 0
#     with torch.no_grad():
#         for data in tqdm(eval_iter):
#             grid_traces = data[0].to(device)
#             tgt_roads = data[1]
#             traces_gps = data[2].to(device)
#             sample_Idx = data[3].to(device)
#             traces_lens, road_lens = data[4], data[5]
#             infer_seq = model.infer(grid_traces=grid_traces,
#                                     traces_gps=traces_gps,
#                                     traces_lens=traces_lens,
#                                     road_lens=road_lens,
#                                     gdata=gdata,
#                                     sample_Idx=sample_Idx,
#                                     tf_ratio=0.)
#             if use_crf:
#                 infer_seq = np.array(infer_seq).flatten()
#             else:
#                 infer_seq = infer_seq.argmax(dim=-1).detach().cpu().numpy().flatten()
#             tgt_roads = tgt_roads.flatten().numpy()
#             mask = (tgt_roads != -1)
#             acc = accuracy_score(infer_seq[mask], tgt_roads[mask])
#             eval_acc_sum += acc
#             count += 1
#     return eval_acc_sum / count, eval_acc_sum / count

def longest_common_subsequence(seq1, seq2):
    # 创建一个二维数组，用于存储两个序列的LCS长度
    lcs_matrix = [[0 for _ in range(len(seq2)+1)] for _ in range(len(seq1)+1)]

    # 动态规划填充这个矩阵
    for i in range(1, len(seq1)+1):
        for j in range(1, len(seq2)+1):
            if seq1[i-1] == seq2[j-1]:
                lcs_matrix[i][j] = lcs_matrix[i-1][j-1] + 1
            else:
                lcs_matrix[i][j] = max(lcs_matrix[i-1][j], lcs_matrix[i][j-1])

    # 矩阵的最后一个元素包含了LCS的长度
    return lcs_matrix[-1][-1]

def evaluate(model, eval_iter, device, gdata, use_crf):
    model.eval()
    eval_acc_sum, count = 0., 0
    with torch.no_grad():
        for data in tqdm(eval_iter):
            grid_traces = data[0].to(device)
            tgt_roads = data[1]
            traces_gps = data[2].to(device)
            sample_Idx = data[3].to(device)
            traces_lens, road_lens = data[4], data[5]
            infer_seq = model.infer(grid_traces=grid_traces,
                                    traces_gps=traces_gps,
                                    traces_lens=traces_lens,
                                    road_lens=road_lens,
                                    gdata=gdata,
                                    sample_Idx=sample_Idx,
                                    tf_ratio=0.)
            if use_crf:
                infer_seq = np.array(infer_seq).flatten()
            else:
                infer_seq = infer_seq.argmax(dim=-1).detach().cpu().numpy().flatten()
            tgt_roads = tgt_roads.flatten().numpy()
            mask = (tgt_roads != -1)
            acc = accuracy_score(infer_seq[mask], tgt_roads[mask])
            eval_acc_sum += acc
            count += 1
    return eval_acc_sum / count



# def main(args):
#     create_dir(f"{args['root_path']}/ckpt/")
#     save_path = "{}/ckpt/bz{}_lr{}_ep{}_edim{}_dp{}_tf{}_tn{}_ng{}_crf{}_wd{}_best.pt".format(
#         args['root_path'], args['batch_size'], args['lr'], args['epochs'],
#         args['emb_dim'], args['drop_prob'], args['tf_ratio'], args['topn'],
#         args['neg_nums'], args['use_crf'], args['wd'])
#     root_path = args['root_path']
#
#     if args['downsample_rate'] == 1.0:
#         downsample_rate = 1
#     else:
#         downsample_rate = args['downsample_rate']
#     data_path = osp.join(args['root_path'], 'data'+str(downsample_rate) + '/')
#     trainset = MyDataset(root_path=root_path, path=data_path, name="train")
#     valset = MyDataset(root_path=root_path, path=data_path, name="val")
#     testset = MyDataset(root_path=root_path, path=data_path, name="test")
#     train_iter = DataLoader(dataset=trainset,
#                             batch_size=args['batch_size'],
#                             shuffle=True,
#                             collate_fn=padding)
#     val_iter = DataLoader(dataset=valset,
#                           batch_size=args['eval_bsize'],
#                           collate_fn=padding)
#     test_iter = DataLoader(dataset=testset,
#                            batch_size=args['eval_bsize'],
#                            collate_fn=padding)
#     print("loading dataset finished!")
#     device = torch.device(f"cuda:{args['dev_id']}" if torch.cuda.is_available() else "cpu")
#     gdata = GraphData(root_path=root_path,
#                       data_path=data_path,
#                       layer=args['layer'],
#                       gamma=args['gamma'],
#                       device=device)
#     print('get graph extra data finished!')
#     model = GMM(emb_dim=args['emb_dim'],
#                 target_size=gdata.num_roads,
#                 topn=args['topn'],
#                 neg_nums=args['neg_nums'],
#                 device=device,
#                 use_crf=args['use_crf'],
#                 bi=args['bi'],
#                 atten_flag=args['atten_flag'],
#                 drop_prob=args['drop_prob'])
#     model = model.to(device)
#     best_acc = 0
#     best_model = None
#     print("loading model finished!")
#     optimizer = optim.AdamW(params=model.parameters(),
#                             lr=args['lr'],
#                             weight_decay=args['wd'])
#     # start training
#     for e in range(args['epochs']):
#         print(f"================Epoch: {e + 1}================")
#         start_time = time.time()
#         train_avg_loss = train(model, train_iter, optimizer, device, gdata, args)
#         end_time = time.time()
#         print("train time: {}".format(end_time - start_time))
#         val_acc, val_rlcs = evaluate(model, val_iter, device, gdata, args['use_crf'], is_rlcs=False)
#         # choose model based on val_acc
#         if best_acc < val_acc:
#             best_model = deepcopy(model)
#             best_acc = val_acc
#         print("Epoch {}: train_avg_loss {} val_avg_acc: {} val_avg_rlcs: {}".format(e + 1, train_avg_loss, val_acc, val_rlcs))
#         nni.report_intermediate_result(val_acc)
#
#     test_acc, test_rlcs = evaluate(best_model, test_iter, device, gdata, args['use_crf'], is_rlcs=True)
#     nni.report_final_result(test_acc)
#     print(f"test_avg_acc: {test_acc:.4f} test_avg_rlcs: {test_rlcs:.4f}")
#     torch.save(best_model.state_dict(), save_path)




def main(args):
    num_clients = 15
    city = 'beijing'
    create_dir(f"{args['root_path']}/ckpt/")
    save_path = "{}/ckpt/bz{}_lr{}_ep{}_edim{}_dp{}_tf{}_tn{}_ng{}_crf{}_wd{}_best.pt".format(
        args['root_path'], args['batch_size'], args['lr'], args['epochs'],
        args['emb_dim'], args['drop_prob'], args['tf_ratio'], args['topn'],
        args['neg_nums'], args['use_crf'], args['wd'])


    if args['downsample_rate'] == 1.0:
        downsample_rate = 1
    else:
        downsample_rate = args['downsample_rate']
    # data_path = osp.join(args['root_path'], 'data' + str(downsample_rate) + '/')
    root_path = osp.join('./data/' + city + '/')
    data_path = osp.join('./data/' + city + '/output_15' + '/')
    device = torch.device(f"cuda:{args['dev_id']}" if torch.cuda.is_available() else "cpu")

    clients_data = []
    models = []
    optimizers = []
    gdata = GraphData(root_path=root_path,
                      data_path=data_path,
                      layer=args['layer'],
                      gamma=args['gamma'],
                      device=device)
    for client_id in tqdm(range(num_clients)):
        client_data_path = osp.join(data_path, f'client_{client_id}/')

        train_dataset = FederatedDataset(data_path, client_data_path, "train", city)
        valid_dataset = FederatedDataset(data_path, client_data_path, "val", city)
        test_dataset = FederatedDataset(data_path, client_data_path, "test", city)

        train_loader = DataLoader(train_dataset, batch_size=args['batch_size'], shuffle=False, collate_fn=padding)
        valid_loader = DataLoader(valid_dataset, batch_size=args['batch_size'], shuffle=False, collate_fn=padding)
        test_loader = DataLoader(test_dataset, batch_size=args['batch_size'], shuffle=False, collate_fn=padding)

        clients_data.append((train_loader, valid_loader, test_loader))


        print('get graph extra data finished!')
        model = GMM(emb_dim=args['emb_dim'],
                    target_size=gdata.num_roads,
                    topn=args['topn'],
                    neg_nums=args['neg_nums'],
                    device=device,
                    use_crf=args['use_crf'],
                    bi=args['bi'],
                    atten_flag=args['atten_flag'],
                    drop_prob=args['drop_prob'])
        model = model.to(device)

        optimizer = optim.Adam(model.parameters(), lr=args['lr'])
        models.append(model)
        optimizers.append(optimizer)

    print("Loading dataset finished!")

    # start training
    # start training
    # -------------------- FedAvg 训练开始 --------------------
    # 这里假设先用第 0 个客户端的模型当做初始全局模型
    global_model = deepcopy(models[0])
    best_model = None
    best_acc = -1.0

    def fedavg(local_models):
        """
        将所有客户端的模型参数做平均，返回新的参数字典。
        如果需要按客户端数据量进行加权，请自行在此函数中修改。
        """
        new_params = OrderedDict()
        # 以第一个模型的 state_dict 为基准
        for k in local_models[0].state_dict().keys():
            # 所有客户端对应参数的平均
            new_params[k] = sum([m.state_dict()[k] for m in local_models]) / len(local_models)
        return new_params

    for e in range(args['epochs']):
        print(f"================Epoch: {e + 1}================")
        start_time = time.time()

        # 下发全局模型给每个客户端，客户端进行本地训练
        for client_id in range(num_clients):
            # 把全局参数同步到本地模型
            # models[client_id].load_state_dict(global_model.state_dict())
            # 用对应客户端的 train_loader 做本地训练
            train_avg_loss = train(
                models[client_id],
                clients_data[client_id][0],  # train_loader
                optimizers[client_id],
                device,
                gdata,
                args,
                client_id
            )

        # 聚合所有客户端模型，更新全局模型
        # new_global_params = fedavg(models)
        # global_model.load_state_dict(new_global_params)
        # print("Aggregating models...")


        end_time = time.time()
        print("train time: {:.4f} s".format(end_time - start_time))

        # 在验证集上评估全局模型（你也可以改成评估各客户端的本地模型）
        val_acc_list = []
        # val_rlcs_list = []
        for client_id in range(num_clients):
            acc_c = evaluate(
                models[client_id], # global_model
                clients_data[client_id][1],  # valid_loader
                device,
                gdata,
                args['use_crf']
            )
            val_acc_list.append(acc_c)
            # val_rlcs_list.append(rlcs_c)

        val_acc = sum(val_acc_list) / len(val_acc_list)
        # val_rlcs = sum(val_rlcs_list) / len(val_rlcs_list)

        # 选择最优模型
        if best_acc < val_acc:
            best_model = deepcopy(models) # global_model
            best_acc = val_acc

        print("Epoch {}: val_avg_acc: {:.4f}".format(
            e + 1, val_acc
        ))
        nni.report_intermediate_result(val_acc)

    # 在测试集上评估最优的全局模型
    test_acc_list = []
    # test_rlcs_list = []
    for client_id in range(num_clients):
        acc_c = evaluate(
            best_model[client_id],  # best_model no index
            clients_data[client_id][2],  # test_loader
            device,
            gdata,
            args['use_crf']
        )
        test_acc_list.append(acc_c)
        # test_rlcs_list.append(rlcs_c)

    test_acc = sum(test_acc_list) / len(test_acc_list)
    # test_rlcs = sum(test_rlcs_list) / len(test_rlcs_list)

    nni.report_final_result(test_acc)
    print(f"test_avg_acc: {test_acc:.4f}")
    # torch.save(best_model.state_dict(), save_path)
    # -------------------- FedAvg 训练结束 --------------------


if __name__ == "__main__":
    try:
        tuner_params = nni.get_next_parameter()
        if tuner_params.get('tf_ratio') and tuner_params['tf_ratio'] == 0:
            tuner_params['tf_ratio'] = 0.0
        if tuner_params.get('drop_prob') and tuner_params['drop_prob'] == 0:
            tuner_params['drop_prob'] = 0.0
        params = vars(merge_parameter(get_params(), tuner_params))
        print(params)
        main(params)
    except Exception as exception:
        raise