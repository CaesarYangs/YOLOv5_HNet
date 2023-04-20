import os
import shutil
import copy
import numpy as np
import torch
from FLCore.FedAvgoptions import args_parser

'''
python FL_Val.py --resume_epoch 21 --num_users 2 --max_epochs 30
python FL_Val.py --max_epochs 10 --num_users 2
'''
args = args_parser()  # parse args
epoch = 20
# idxs_users = np.arange(args.num_users) + 1  # 客户端列表
idxs_users = [1, 3, 4]
# for epoch in range(args.resume_epoch, args.max_epochs + 1):  # 对全局的每个训练周期：
#     for idx in idxs_users:  # 对每个参与联邦训练的客户端。
#         print("Rounds:{}, client{}  val:".format(epoch, idx))
#         # file_old_name = "datasets{}".format(idx)  # 数据集改名
#         # file_new_name = "datasets"
#         # os.rename(file_old_name, file_new_name)
#         os.system("python val.py  --data data/fl_clients/{}.yaml --weights weights/avg_ckpt_E{}.pt".format(idx, epoch))
#         exp_name = "runs/val/exp"  # exp改名
#         exp_new_name = "runs/val/E{}_C{}".format(epoch, idx)
#         os.rename(exp_name, exp_new_name)
#         # os.rename(file_new_name, file_old_name)  # val完要把数据集名字改回去

for idx in idxs_users:  # 对每个参与联邦训练的客户端。
    print("client{}  val:".format(idx))
    # file_old_name = "datasets{}".format(idx)  # 数据集改名
    # file_new_name = "datasets"
    # os.rename(file_old_name, file_new_name)
    os.system("python val.py  --data data/fl_clients/{}.yaml --weights /Users/caesaryang/Developer/1-Graduate/res_analysis/fl/FL-E3-baseline_3/weights/avg_ckpt_E{}.pt".format(idx, epoch))
    exp_name = "runs/val/exp"  # exp改名
    exp_new_name = "runs/val/E{}_C{}".format(epoch, idx)
    os.rename(exp_name, exp_new_name)
    # os.rename(file_new_name, file_old_name)  # val完要把数据集名字改回去
