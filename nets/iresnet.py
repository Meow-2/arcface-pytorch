
import torch
from torch import nn

# 用于指定 python 模块中的哪些模块应该被导入
__all__ = ['iresnet18', 'iresnet34', 'iresnet50', 'iresnet100', 'iresnet200']


# W/H=[(输入大小-卷积核大小+2*P）/步长]  +1.

# in_channels：输入信号的通道数。这是卷积核需要考虑的“深度”。
# out_channels：输出信号的通道数。这是卷积核生成的输出信号的“深度”。
# kernel_size：卷积核的尺寸。可以是一个数字（如 3）或者一个包含两个数字的序列（如（3，3））。如果是数字，则表示卷积核是正方形的。
# stride：卷积的步幅。可以是一个数字（如 1）或者一个包含两个数字的序列（如（1，1））。如果是数字，则表示在每一维上的步幅相同。
# padding：填充的数量。可以是一个数字（如 1）或者一个包含两个数字的序列（如（1，1））。如果是数字，则表示在每一维上的填充数量相同。
# dilation：卷积核的膨胀系数。可以是一个数字（如 1）或者一个包含两个数字的序列（如（1，1））。如果是数字，则表示在每一维上的膨胀系数相同。
# groups：分组卷积的组数。卷积核的输入和输出通道将被分成若干组。默认值为 1。
# bias: 是否使用偏置项。如果设置为 True，则每个输出通道都将有一个独立的偏置项。默认值为 True

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    return nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=3,
                     stride=stride,
                     padding=dilation,
                     groups=groups,
                     bias=False,
                     dilation=dilation)


# 没有 padding, 下采样输出大小和 3x3 一致
def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)

# 这里的 3x3 和 conv1x1 卷积都不会改变图片的大小, 因为 3x3 有 padding = 1, 所以不变


# 残差块
class IBasicBlock(nn.Module):
    # 通过类名调用 IBasicBlock.expansion 得到的是静态变量, 这里有坑
    expansion = 1
    # base_width 就没有生效, groups 也没有生效

    def __init__(self, inplanes, planes, stride=1, downsample=None,
                 groups=1, base_width=64, dilation=1):
        super(IBasicBlock, self).__init__()
        if groups != 1 or base_width != 64:
            raise ValueError(
                'BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError(
                "Dilation > 1 not supported in BasicBlock")
        # eps为稳定系数
        self.bn1 = nn.BatchNorm2d(inplanes, eps=1e-05,)  # 不改变通道数和图片尺寸
        self.conv1 = conv3x3(inplanes, planes)  # 改变通道数, 不改变图片尺寸
        self.bn2 = nn.BatchNorm2d(planes, eps=1e-05,)  # 不改变通道数和图片尺寸
        self.prelu = nn.PReLU(planes)  # 不改变通道数和图片尺寸
        # 第二个卷积的 stride 可以调整, 如果有下采样步骤的话, 就会在这里下采样
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn3 = nn.BatchNorm2d(planes, eps=1e-05,)  # 不改变通道数和图片尺寸
        self.downsample = downsample  # 下采样并调整通道数以及 BN
        self.stride = stride

        # nn.PReLU(
        #     num_parameters: int = 1,
        #     init: float = 0.25,
        #     device=None,
        #     dtype=None,
        # )

    # 如果调用时没有下采样这一步, inplanes 和 planes 一定要相等
    # 如果 stride = 1, 就没有下采样, 调用时应保证通道不变
    # 如果 stride = 2, 就有下采样, indentify 也要进行下采样, 调用时, 通道会变为 2 倍, indentify 也要变两倍
    def forward(self, x):
        # 记录下当前的 x
        identity = x
        out = self.bn1(x)
        out = self.conv1(out)  # 通道数变了
        out = self.bn2(out)
        out = self.prelu(out)
        out = self.conv2(out)  # 可能改变图像大小
        out = self.bn3(out)

        # 注意这里的输入是x, 所以总共只进行了一次下采样
        if self.downsample is not None:
            identity = self.downsample(x)  # 使用 1x1 的卷积进行下采样
        out += identity
        return out


# IResNet(IBasicBlock, [2,2,2,2],dropout_keep_prob=0.5, embedding_size=embedding_size)
class IResNet(nn.Module):
    fc_scale = 7 * 7

    def __init__(self,
                 block, layers, dropout_keep_prob=0, embedding_size=512, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None, fp16=False):
        super(IResNet, self).__init__()
        self.fp16 = fp16
        self.inplanes = 64
        self.dilation = 1

        # replace_stride_with_dilation 就是 [False , False , False]
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups  # 1
        self.base_width = width_per_group  # 64, 每个 group 的宽度, 没有生效, 没什么用

        # 输入信号的通道数为 3 , 输出 inplanes 的通道数为 64 ,图片尺寸不变
        self.conv1 = nn.Conv2d(
            3, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)
        # BN
        self.bn1 = nn.BatchNorm2d(self.inplanes, eps=1e-05)
        # 激活
        self.prelu = nn.PReLU(self.inplanes)
        # 卷积 -> BatchNorm2d —> PReLU
        self.layer1 = self._make_layer(block, 64, layers[0], stride=2)
        self.layer2 = self._make_layer(block,
                                       128,
                                       layers[1],
                                       stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block,
                                       256,
                                       layers[2],
                                       stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block,
                                       512,
                                       layers[3],
                                       stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.bn2 = nn.BatchNorm2d(512 * block.expansion, eps=1e-05,)
        # dropout_keep_prob = 0.5, 用来防止过拟合, 百分之五十概率将输入张量中的元素置为零
        # inplace 代表在原张量上替换
        self.dropout = nn.Dropout(
            p=dropout_keep_prob, inplace=True)  # dropout 层
        # block.expansion = 1, self.fc_scale = 7, embedding_size = 512
        self.fc = nn.Linear(512 * block.expansion *
                            self.fc_scale, embedding_size)
        # BN
        self.features = nn.BatchNorm1d(embedding_size, eps=1e-05)
        nn.init.constant_(self.features.weight, 1.0)
        self.features.weight.requires_grad = False

        # 初始化模型权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # 从给定均值和标准差的正态分布N(0, 0.1)中生成值来初始化
                nn.init.normal_(m.weight, 0, 0.1)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                # 初始化为常量
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # 不会生效
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, IBasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    # block = IBasicBlock , 输出通道数64->128->256->512, layers[0]->[1]—>[2]->[3], stride = 2, dilate = False
    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        downsample = None
        previous_dilation = self.dilation  # 1
        # 全部是 False , 不会生效
        if dilate:
            self.dilation *= stride
            stride = 1
        # 如果 stride 不为 1 , 或者最开始的输入通道 inplanes = 64 和 layer 的 planes 不一样的话就进行下采样

        # 因为 stride 全部是 2, 所以一定会生效
        if stride != 1 or self.inplanes != planes * block.expansion:  # block.expansion = 1
            downsample = nn.Sequential(
                # 使用 1x1 卷积调整通道数, 将 inplanes 调整成 planes * block.expansion, stride = 2
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion, eps=1e-05, ),
            )
        layers = []
        layers.append(
            block(self.inplanes, planes, stride, downsample, self.groups,
                  self.base_width, previous_dilation))
        self.inplanes = planes * block.expansion  # 将输入通道数改变成输出的通道数
        # range [左闭, 右开)
        # 不会下采样, 也不会改变通道数
        for _ in range(1, blocks):
            layers.append(
                block(self.inplanes,    # stride 为 1, 不改变尺寸
                      planes,
                      groups=self.groups,  # 1
                      base_width=self.base_width,  # 64, 没有什么用
                      dilation=self.dilation))  # 1

        return nn.Sequential(*layers)  # 组合成一个 layers

    # input (64, 3, 112, 112)
    def forward(self, x):  # output:
        x = self.conv1(x)  # (64, 64, 112, 112)
        x = self.bn1(x)    # (64, 64, 112, 112)
        x = self.prelu(x)  # (64, 64, 112, 112)
        x = self.layer1(x)  # (64, 64, 56, 56)  , 两个残差模块
        x = self.layer2(x)  # (64, 128, 28, 28) , 两个残差模块
        x = self.layer3(x)  # (64, 256, 14, 14) , 两个残差模块
        x = self.layer4(x)  # (64, 512, 7, 7) , 两个残差模块
        x = self.bn2(x)
        # 从第二维开始平坦化, 也就是从第二维开始拉平
        x = torch.flatten(x, 1)  # (64, 512x7x7)
        # 随机 dropout
        x = self.dropout(x)  # (64, 512x7x7)
        x = self.fc(x)  # (64, 512)
        x = self.features(x)  # (64, 512)
        return x


# 没有用到 arch 和 progress 和 pretrained
def _iresnet(arch, block, layers, pretrained, progress, **kwargs):
    # 实际上调用的模型 IResNet(IBasicBlock, [2,2,2,2],dropout_keep_prob=0.5, embedding_size=embedding_size)
    model = IResNet(block, layers, **kwargs)
    if pretrained:
        raise ValueError("No pretrained model for iresnet")
    return model


# 调用的时候传入的参数
# self.arcface = iresnet18(dropout_keep_prob=0.5, embedding_size=embedding_size, pretrained=pretrained)
# pretrained 使用的是 train.py 中设置的 pretrained
# 额外传入的参数有 dropout_keep_prob=0.5, embedding_size=embedding_size
def iresnet18(pretrained=False, progress=True, **kwargs):
    # block = IBasicBlock -> 传给 IResNet
    # layers = [2,2,2,2] -> 传给 IResNet, 2 代表残差块的数量, 第一个残差块会改变通道数并下采样, 而剩余的不会
    return _iresnet('iresnet18', IBasicBlock, [2, 2, 2, 2], pretrained,
                    progress, **kwargs)


def iresnet34(pretrained=False, progress=True, **kwargs):
    return _iresnet('iresnet34', IBasicBlock, [3, 4, 6, 3], pretrained,
                    progress, **kwargs)


def iresnet50(pretrained=False, progress=True, **kwargs):
    return _iresnet('iresnet50', IBasicBlock, [3, 4, 14, 3], pretrained,
                    progress, **kwargs)


def iresnet100(pretrained=False, progress=True, **kwargs):
    return _iresnet('iresnet100', IBasicBlock, [3, 13, 30, 3], pretrained,
                    progress, **kwargs)


def iresnet200(pretrained=False, progress=True, **kwargs):
    return _iresnet('iresnet200', IBasicBlock, [6, 26, 60, 6], pretrained,
                    progress, **kwargs)
