#!/bin/sh

# 跑5遍
# for i in {0..3}; do
#     python train.py --dataset SCface_train
# done

# for i in {0..1}; do
#     python train.py --dataset lfw2_no_triplet --epoch 100
# done

# for i in {0..1}; do
#     python train.py --dataset lfw_no_triplet_1 --epoch 150
# done
#
# for i in {0..2}; do
#     # echo $i
#     python train.py --dataset lfw --epoch 150
# done

for i in {0..2}; do
    # echo $i
    python train.py --dataset lfw_SCface_train --epoch 100
done
