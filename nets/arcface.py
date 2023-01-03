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
        # s 是一个缩放用的参数
        self.s = s
        # m 是角度偏置
        # m: additive angular margin
        self.m = m
        # torch.FloatTensor(), 创建一个浮点数数组 (num_classes, embedding_size)
        # Parameter对象作为模型的一部分的时候, 可以被自动求导
        self.weight = Parameter(torch.FloatTensor(num_classes, embedding_size))
        # 使用了 Xavier 均匀分布来随机初始化权重, 有助于网络更快地收敛，并且可以在训练过程中较少地出现梯度消失或爆炸的问题
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(m)    # 固定值
        self.sin_m = math.sin(m)    # 固定值
        # self.th = - self.cos_m
        self.th = math.cos(math.pi - m)     # 固定值
        # self.mm = self.sin_m*m
        self.mm = math.sin(math.pi - m) * m  # 固定值

        # input (64,512) label(64,1)
    def forward(self, input, label):
        # 在计算图之外进行运算, y = F.linear(x,A,b), 也就是说这个线性变换的操作不是可学习的, A 矩阵是指定的权重矩阵, 不指定b, 则 b为零矩阵
        # F.normalize(self.weight, p = 2, dim = 1) 按 L2 范数, 第1维进行标准化
        # 因为进行了归一化, 所以得到的 cos 的值
        # 输出的是一个 (num_class) 维的向量, 每一个类别上是 cos 值, 真实标签的那个类别就是真实 cos
        # F.linear 是可以处理 batch 的, 他会对每个样本进行线性变换, 最后再拼接
        # cosine (64, num_classes)
        cosine = F.linear(input, F.normalize(self.weight))
        # 对矩阵每一元素求其 sin 值, torch.clamp(0,1) 用于将矩阵的元素裁剪到 (0,1), 防止溢出
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        # 角度偏置 cos(西塔+m); cos西塔 * cosm - sin西塔 * sinm
        phi = cosine * self.cos_m - sine * self.sin_m
        # torch.where(condition,x,y) 当 condition 为真时, 返回 x, 不然返回 y
        # 如果那一个类别的 cos西塔 > - cos m , 就变为 cos(西塔 + m), 不然就变成 cos 西塔- m*sinm
        # cos(theta + m) 会使 theta 变小, 从而进一步抑制
        # 而角间距在临近 180 度的时候会有溢出的问题，原来的角度再加上一个间距，可能会超过 180 度
        # 这时候就不满足函数单调的性质，这时可以 drop to cosface
        phi = torch.where(cosine.float() > self.th,
                          phi.float(), cosine.float() - self.mm)
        # 理解为引入了一个恒定的角度阈值来降低高置信噪声数据
        # ========================================================================
        # if self.easy_margin:
        #     cosine>0 means two class is similar, thus use the phi which make it
        #     phi = torch.where(cosine > 0, phi, cosine)
        # else:
        #     phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        # ========================================================================
        # one_hot (64, num_classes)

        # 这些损失为什么有效果
        # https://zhuanlan.zhihu.com/p/103766001
        one_hot = torch.zeros(cosine.size()).type_as(phi).long()
        # lable (64,1)
        # 将 one_hot 张量的第 1 维的第 label.view(-1, 1).long() 个元素都赋值为 1
        # scatter_(dim=1, index , 赋值为1)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        # 只有 one_hot 为1的地方是新的值, one_hot 为 0 的地方都是原值
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
