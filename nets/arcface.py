import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module, Parameter

from nets.iresnet import (iresnet18, iresnet34, iresnet50, iresnet100,
                          iresnet200)
from nets.mobilefacenet import get_mbf
from nets.mobilenet import get_mobilenet

#self.head = Arcface_Head(embedding_size=embedding_size, num_classes=num_classes, s=s)


class Arcface_Head(Module):
    def __init__(self, embedding_size=128, num_classes=10575, s=64., m=0.5):
        super(Arcface_Head, self).__init__()
        self.s = s
        self.m = m
        # torch.FloatTensor(), 创建一个浮点数数组 (num_classes, embedding_size)
        # Parameter对象作为模型的一部分的时候, 可以被自动求导
        self.weight = Parameter(torch.FloatTensor(num_classes, embedding_size))
        # 使用了 Xavier 均匀分布来随机初始化权重, 有助于网络更快地收敛，并且可以在训练过程中较少地出现梯度消失或爆炸的问题
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(m)    # 固定值
        self.sin_m = math.sin(m)    # 固定值
        self.th = math.cos(math.pi - m)     # 固定值
        self.mm = math.sin(math.pi - m) * m  # 固定值

    def forward(self, input, label):
        # 在计算图之外进行运算, y = F.linear(x,A,b)
        cosine = F.linear(input, F.normalize(self.weight))
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine.float() > self.th,
                          phi.float(), cosine.float() - self.mm)

        one_hot = torch.zeros(cosine.size()).type_as(phi).long()
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.s
        return output


class Arcface(nn.Module):
    def __init__(self, num_classes=None, backbone="mobilefacenet", pretrained=False, mode="train"):
        super(Arcface, self).__init__()
        if backbone == "mobilefacenet":
            embedding_size = 128
            s = 32
            self.arcface = get_mbf(
                embedding_size=embedding_size, pretrained=pretrained)

        elif backbone == "mobilenetv1":
            embedding_size = 512
            s = 64
            self.arcface = get_mobilenet(
                dropout_keep_prob=0.5, embedding_size=embedding_size, pretrained=pretrained)

        elif backbone == "iresnet18":
            embedding_size = 512
            s = 64
            # 实际上调用的模型 IResNet(IBasicBlock, [2,2,2,2],dropout_keep_prob=0.5, embedding_size=embedding_size, )
            self.arcface = iresnet18(
                dropout_keep_prob=0.5, embedding_size=embedding_size, pretrained=pretrained)

        elif backbone == "iresnet34":
            embedding_size = 512
            s = 64
            self.arcface = iresnet34(
                dropout_keep_prob=0.5, embedding_size=embedding_size, pretrained=pretrained)

        elif backbone == "iresnet50":
            embedding_size = 512
            s = 64
            self.arcface = iresnet50(
                dropout_keep_prob=0.5, embedding_size=embedding_size, pretrained=pretrained)

        elif backbone == "iresnet100":
            embedding_size = 512
            s = 64
            self.arcface = iresnet100(
                dropout_keep_prob=0.5, embedding_size=embedding_size, pretrained=pretrained)

        elif backbone == "iresnet200":
            embedding_size = 512
            s = 64
            self.arcface = iresnet200(
                dropout_keep_prob=0.5, embedding_size=embedding_size, pretrained=pretrained)
        else:
            raise ValueError(
                'Unsupported backbone - `{}`, Use mobilefacenet, mobilenetv1.'.format(backbone))

        self.mode = mode
        if mode == "train":
            self.head = Arcface_Head(
                embedding_size=embedding_size, num_classes=num_classes, s=s)

    def forward(self, x, y=None, mode="predict"):
        x = self.arcface(x)  # 返回一个一维的向量  (64, 512)
        x = x.view(x.size()[0], -1)  # 拉成一维
        # 在计算图之外进行标准化
        x = F.normalize(x)  # 标准化
        if mode == "predict":
            return x
        else:
            x = self.head(x, y)
            return x
