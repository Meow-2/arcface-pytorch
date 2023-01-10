#!/bin/sh

# 跑5遍
for i in {0..3}; do
    # echo $i
    python train.py --dataset lfw_no_triplet_1
done

for i in {0..3}; do
    # echo $i
    python train.py --dataset lfw_triplet_1
done

for i in {0..3}; do
    # echo $i
    python train.py --dataset lfw
done
