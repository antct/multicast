#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python main.py --model_name multicast --bag_size 3 --pretrain_epoch 0 --pretrain_lr 0.5 --lr 0.1 --max_epoch 200 --seed $1
