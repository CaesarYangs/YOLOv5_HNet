#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Python version: 3.6
# python FedAvg_main.py --num_users 2 --epochs 5
import argparse


def args_parser():
    parser = argparse.ArgumentParser()
    # federated arguments
    parser.add_argument('--max_epochs', type=int,
                        default=20, help="rounds of training")
    parser.add_argument('--num_users', type=int, default=4,
                        help="number of users: K")
    parser.add_argument('--local_epochs', type=int,
                        default=5, help="local epochs of training")
    parser.add_argument('--local_epochs_fast', type=int,
                        default=3, help="local epochs of training")
    parser.add_argument('--local_epochs_slow', type=int,
                        default=5, help="local epochs of training")
    parser.add_argument('--resume_epoch', type=int, default=1,
                        help="resume global epochs of communication")  # 断点（在第x轮通信中断）

    args = parser.parse_args()
    return args
