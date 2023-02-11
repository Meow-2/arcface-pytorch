#!/bin/sh

# for i in {0..2}; do
#     python train.py --dataset lfw_SCface_train --epoch 100
# done

# for i in {0..1}; do
#     python train.py --dataset lfw_no_triplet_1 --epoch 150
# done

for i in {0..2}; do
    python train.py --dataset lfw --epoch 100 --model "/home/zk/project/arcface-pytorch/model_data/ms1mv3_iresnet50.pth"
done
#
# for i in {0..2}; do
#     python train.py --dataset lfw_no_triplet_1 --epoch 100
# done
#
# for i in {0..2}; do
#     python train.py --dataset lfw --epoch 100
# done
