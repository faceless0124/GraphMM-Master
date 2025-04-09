import nni
import numpy as np
import torch
import torch.optim as optim
from config import get_params
from nni.utils import merge_parameter
from model.gmm import GMM
from data_loader import MyDataset, padding
from torch.utils.data import DataLoader
from graph_data import GraphData
from tqdm import tqdm
import torch.nn as nn
from sklearn.metrics import accuracy_score
import os.path as osp
from copy import deepcopy
from data_preprocess.utils import create_dir
import time

def train(model, train_iter, optimizer, device, gdata, args):
    model.train()
    train_l_sum, count = 0., 0
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
        if count % 5 == 0:
            print(f"Iteration {count}: train_loss {loss.item()}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.)
        optimizer.step()
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

def evaluate(model, eval_iter, device, gdata, use_crf, is_rlcs):
    model.eval()
    eval_acc_sum, count = 0., 0
    eval_rlcs_sum = 0.
    all_time = 0.
    with torch.no_grad():
        for data in tqdm(eval_iter):
            grid_traces = data[0].to(device)
            tgt_roads = data[1]
            traces_gps = data[2].to(device)
            sample_Idx = data[3].to(device)
            traces_lens, road_lens = data[4], data[5]

            updated_traces_lens = torch.zeros(len(traces_lens), dtype=int)
            updated_roads_lens = torch.zeros(len(road_lens), dtype=int)

            # 根据sample_idx处理tgt_road
            sample_Idx_masked = torch.where(sample_Idx == -1, 0, sample_Idx).cpu()
            result = tgt_roads[torch.arange(tgt_roads.size(0)).unsqueeze(1), sample_Idx_masked]
            result[sample_Idx == -1] = -1
            result = result.to(device)
            match_interval = 4
            # 初始化用于收集infer_seq的列表
            collected_infer_seq = [[] for _ in range(0, grid_traces.size(0))]
            last = 0

            for i in range(match_interval - 1, grid_traces.size(1) - match_interval + 1, match_interval):
                last = i
                for j in range(len(traces_lens)):
                    updated_traces_lens[j] = i + 1
                    updated_roads_lens[j] = i + 1
                # incremental
                start_time = time.time()
                infer_seq = model.infer(grid_traces=grid_traces[:, :i + 1],
                                        traces_gps=traces_gps[:, :i + 1],
                                        traces_lens=updated_traces_lens,
                                        road_lens=updated_roads_lens,
                                        gdata=gdata,
                                        sample_Idx=sample_Idx[:, :i + 1],
                                        tf_ratio=0.)
                end_time = time.time()
                all_time += end_time - start_time
                # 收集infer_seq中每个元素的最后一个元素
                if use_crf:
                    for i, seq in enumerate(infer_seq):
                        collected_infer_seq[i].extend(seq[-match_interval:])
                else:
                    collected_infer_seq.extend([seq.argmax(dim=-1).detach().cpu().numpy()[-1] for seq in infer_seq])

            for j in range(len(traces_lens)):
                updated_traces_lens[j] = grid_traces.size(1)-last-1
                updated_roads_lens[j] = grid_traces.size(1)-last-1
            start_time = time.time()
            infer_seq = model.infer(grid_traces=grid_traces[:, last + 1:],
                                    traces_gps=traces_gps[:, last + 1:],
                                    traces_lens=updated_traces_lens,
                                    road_lens=updated_roads_lens,
                                    gdata=gdata,
                                    sample_Idx=sample_Idx[:, last + 1:],
                                    tf_ratio=0.)
            end_time = time.time()
            all_time += end_time - start_time
            # 收集infer_seq中每个元素的最后一个元素
            if use_crf:
                for i, seq in enumerate(infer_seq):
                    collected_infer_seq[i].extend(seq)
            else:
                collected_infer_seq.extend([seq.argmax(dim=-1).detach().cpu().numpy()[-1] for seq in infer_seq])

            infer_seqs = np.array(collected_infer_seq)
            result_seqs = result.cpu().numpy()
            r_lcs_values = []
            acc_values = []
            for infer_seq, result_seq in zip(infer_seqs, result_seqs):
                # 应用mask以过滤掉值为-1的元素
                mask = result_seq != -1

                filtered_infer_seq = infer_seq[mask]
                filtered_result_seq = result_seq[mask]

                if is_rlcs:
                    lcs_length = longest_common_subsequence(filtered_infer_seq, filtered_result_seq)
                else:
                    lcs_length = 0

                r_lcs_values.append(lcs_length / len(filtered_infer_seq))

                acc_values.append((filtered_infer_seq == filtered_result_seq).sum() / len(filtered_infer_seq))


            # 计算平均R-LCS
            average_r_lcs = sum(r_lcs_values) / len(r_lcs_values)
            eval_rlcs_sum += average_r_lcs
            average_acc = sum(acc_values) / len(acc_values)
            eval_acc_sum += average_acc


            # if use_crf:
            #     infer_seq_combined = np.array(collected_infer_seq).flatten()
            # else:
            #     infer_seq_combined = np.array(collected_infer_seq)
            #
            # result = result.flatten().cpu().numpy()
            # mask = (result != -1)
            # acc = accuracy_score(infer_seq_combined[mask], result[mask])
            # eval_acc_sum += acc
            count += 1
    print("all time: {}".format(all_time))

    return eval_acc_sum / count, eval_rlcs_sum / count


def main(args):
    create_dir(f"{args['root_path']}/ckpt/")
    save_path = "{}/ckpt/bz{}_lr{}_ep{}_edim{}_dp{}_tf{}_tn{}_ng{}_crf{}_wd{}_best.pt".format(
        args['root_path'], args['batch_size'], args['lr'], args['epochs'],
        args['emb_dim'], args['drop_prob'], args['tf_ratio'], args['topn'],
        args['neg_nums'], args['use_crf'], args['wd'])
    root_path = args['root_path']

    if args['downsample_rate'] == 1.0:
        downsample_rate = 1
    else:
        downsample_rate = args['downsample_rate']
    data_path = osp.join(args['root_path'], 'data'+str(downsample_rate) + '/')
    trainset = MyDataset(root_path=root_path, path=data_path, name="train")
    valset = MyDataset(root_path=root_path, path=data_path, name="val")
    testset = MyDataset(root_path=root_path, path=data_path, name="test")
    train_iter = DataLoader(dataset=trainset,
                            batch_size=args['batch_size'],
                            shuffle=True,
                            collate_fn=padding)
    val_iter = DataLoader(dataset=valset,
                          batch_size=args['eval_bsize'],
                          collate_fn=padding)
    test_iter = DataLoader(dataset=testset,
                           batch_size=args['eval_bsize'],
                           collate_fn=padding)
    print("loading dataset finished!")
    device = torch.device(f"cuda:{args['dev_id']}" if torch.cuda.is_available() else "cpu")
    gdata = GraphData(root_path=root_path,
                      data_path=data_path,
                      layer=args['layer'],
                      gamma=args['gamma'],
                      device=device)
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
    best_acc = 0
    best_model = None
    print("loading model finished!")
    optimizer = optim.AdamW(params=model.parameters(),
                            lr=args['lr'],
                            weight_decay=args['wd'])
    # start training
    for e in range(args['epochs']):
        print(f"================Epoch: {e + 1}================")
        start_time = time.time()
        train_avg_loss = train(model, train_iter, optimizer, device, gdata, args)
        end_time = time.time()
        print("train time: {}".format(end_time - start_time))
        val_acc, val_rlcs = evaluate(model, val_iter, device, gdata, args['use_crf'], is_rlcs=False)
        # choose model based on val_acc
        if best_acc < val_acc:
            best_model = deepcopy(model)
            best_acc = val_acc
        print("Epoch {}: train_avg_loss {} val_avg_acc: {} val_avg_rlcs: {}".format(e + 1, train_avg_loss, val_acc, val_rlcs))
        nni.report_intermediate_result(val_acc)

    test_acc, test_rlcs = evaluate(best_model, test_iter, device, gdata, args['use_crf'], is_rlcs=True)
    nni.report_final_result(test_acc)
    print(f"test_avg_acc: {test_acc:.4f} test_avg_rlcs: {test_rlcs:.4f}")
    torch.save(best_model.state_dict(), save_path)


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
