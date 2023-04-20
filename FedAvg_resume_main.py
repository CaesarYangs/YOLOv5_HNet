#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
import os
import shutil
import copy
import numpy as np
import torch
from FLCore.FedAvgoptions import args_parser

'''
参与训练的数据集命名为：datasets1, datasets2, datasets3...
运行voc_label_main.py生成yolov5要求的数据集格式
python FedAvg_resume_main.py --max_epochs 10 --local_epochs 5 --num_users 4
'''


def update_global(local_models):
    mean_state_dict = {}
    global_model = copy.deepcopy(local_models[1])  # copy列表里第0个model，因为要学文件的格式
    for name, param in global_model.state_dict().items():  # 对于全局模型参数的每一块
        vs = []
        for client in local_models.keys():  # local_models是字典，key是客户端编号，value是对应客户端的权重
            vs.append(local_models[client].float().state_dict()[
                      name])  # 把每个客户端的model里的这一块的参数拿出来放列表里
        vs = torch.stack(vs, dim=0)

        try:
            mean_value = vs.mean(dim=0)  # 求平均

        except Exception:
            # for BN's cnt
            mean_value = (1.0 * vs).mean(dim=0).long()
        mean_state_dict[name] = mean_value  # 把算得的各块平均放到模具里

    # print(mean_state_dict)
    global_model.load_state_dict(
        mean_state_dict, strict=False)  # 加载这个聚合模型到global_model
    return global_model


def main():
    args = args_parser()  # parse args
    # idxs_users = np.arange(args.num_users) + 1  # 客户端列表，让编号从1开始
    idxs_users = [1, 3, 4]
    '''联邦训练开始'''
    for epoch in range(args.resume_epoch, args.max_epochs + 1):  # 对全局的每个训练周期：
        local_models = {}  # 收集"model"
        for idx in idxs_users:  # 对每个参与联邦训练的客户端。
            print("client{} is training(Communication Rounds {})".format(idx, epoch))
            # file_old_name = "datasets{}".format(idx)  # 数据集改名
            # file_new_name = "datasets"
            # os.rename(file_old_name, file_new_name)

            # 某个客户端开始训练
            if epoch == 1:  # 如果是首轮训练，使用初始权重。
                os.system(
                    "python train.py --epochs {} --batch 16 --data data/fl_clients/{}.yaml --device '0' --weights weights/yolov5m.pt".format(args.local_epochs, idx))

            else:  # 非首轮训练，使用聚合后的权重。
                os.system(
                    "python train.py --epochs {} --batch 16 --data data/fl_clients/{}.yaml --device '0' --weights weights/avg_ckpt_E{}.pt".format(
                        args.local_epochs, idx, epoch - 1))

            weights_save_path = "runs/train/exp/weights/" + "last.pt"  # 每个客户端训练完后的last_ckpt

            # weights_remove_path = "Client_ckpt/" + "ckpt_E{}_C{}.pt".format(epoch, idx)  # 要复制到的文件夹和新名字
            # shutil.copyfile(weights_save_path, weights_remove_path)  # 复制并改名
            # ckpt_load = torch.load(weights_remove_path, map_location='cpu')  # 加载权重文件
            ckpt_load = torch.load(
                weights_save_path, map_location='cpu')  # 加载权重文件

            ckpt_model = ckpt_load["model"]  # 把model部分拿出来
            # 字典。每个客户端训练出的model都深拷贝到local_models字典里
            local_models[idx] = copy.deepcopy(ckpt_model)

            exp_name = "runs/train/exp"  # exp改名
            exp_new_name = "runs/train/E{}_C{}".format(epoch, idx)
            os.rename(exp_name, exp_new_name)

            # os.rename(file_new_name, file_old_name)  # 练完把数据集名字改回去

        print("FedAvg start(Epochs = {}):".format(epoch))
        global_model = update_global(local_models)  # 联邦平均聚合权重。返回聚合后的model
        print("FedAvg finished.")
        print("Updating avg “model” to ckpt...")
        # 先深拷贝一份ckpt
        copy_ckpt = copy.deepcopy(exp_new_name + '/weights/best.pt')
        # 加载拷贝的ckpt文件
        Avg_ckpt = torch.load(copy_ckpt, map_location='cpu')
        # 把文件里的model改成聚合后的
        Avg_ckpt["model"] = global_model
        # 把改完的ckpt文件存到weights里
        torch.save(Avg_ckpt, "./weights/avg_ckpt_E{}.pt".format(epoch))
        print("”Avg_Ckpt“(Rounds: {}) Save successfully.".format(epoch))


if __name__ == "__main__":
    main()
