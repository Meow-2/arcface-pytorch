import os

import numpy as np
import torch
import torch.utils.data as data
import torchvision.datasets as datasets
from PIL import Image

from .utils import cvtColor, preprocess_input, resize_image


class FacenetDataset(data.Dataset):
    def __init__(self, input_shape, lines, random):
        self.input_shape = input_shape
        self.lines = lines
        self.random = random

    def __len__(self):
        return len(self.lines)

    # 获取[a,b)的随机数
    # np.random.rand() 的区间是[0,1)
    def rand(self, a=0, b=1):
        return np.random.rand()*(b-a) + a

    def __getitem__(self, index):
        # 图片所在路径
        annotation_path = self.lines[index].split(';')[1].split()[0]
        # 标签的值
        y = int(self.lines[index].split(';')[0])

        # 把图片转换为 RGB
        image = cvtColor(Image.open(annotation_path))
        # ------------------------------------------#
        #   翻转图像
        # ------------------------------------------#
        if self.rand() < .5 and self.random:
            # 左右翻转
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
        # 图片 resize
        image = resize_image(
            image, [self.input_shape[1], self.input_shape[0]], letterbox_image=True)

        # 转化为 BCHW
        image = np.transpose(preprocess_input(
            np.array(image, dtype='float32')), (2, 0, 1))
        # 返回图片和标签
        return image, y


def dataset_collate(batch):
    images = []
    targets = []
    for image, y in batch:
        images.append(image)
        targets.append(y)
    # Use torch.from_numpy to create the Tensor instead of torch.Tensor.
    # In this way, the Tensor created will share memory with the numpy array
    images = torch.from_numpy(np.array(images)).type(torch.FloatTensor)
    # tensor.long() 与 self.to(torch.int64) 等价
    targets = torch.from_numpy(np.array(targets)).long()
    return images, targets


class LFWDataset(datasets.ImageFolder):
    def __init__(self, dir, pairs_path, image_size, transform=None):
        super(LFWDataset, self).__init__(dir, transform)
        self.image_size = image_size
        self.pairs_path = pairs_path

        # 里面装的是元组, 即(图一路径, 图二路径, 是否是同一张图)
        self.validation_images = self.get_lfw_paths(dir)

    def read_lfw_pairs(self, pairs_filename):
        pairs = []
        # 读取 pairs.txt 文件
        with open(pairs_filename, 'r') as f:
            # f.readlines()获得的是一个字符串列表, 里面是 pairs.txt 的每一行
            for line in f.readlines()[1:]:  # 舍去列表第一行
                # str.strip()会将字符串的前后空白和换行符删除
                # str.split()按指定字符分割字符串, 不写就是按空格分割, 返回一个列表
                pair = line.strip().split()
                pairs.append(pair)
        return np.array(pairs)

    def get_lfw_paths(self, lfw_dir, file_ext="jpg"):

        # 返回每一行的一个列表表示
        pairs = self.read_lfw_pairs(self.pairs_path)

        nrof_skipped_pairs = 0
        path_list = []  # 里面装的是元组, 即(图一路径, 图二路径, 是否是同一张图)
        issame_list = []  # 每什么用

        # 遍历 pairs 列表
        for i in range(len(pairs)):
            # for pair in pairs:
            pair = pairs[i]  # pair 为一个列表, 他的内容是 pairs.txt的每一行
            if len(pair) == 3:
                path0 = os.path.join(
                    lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[1])+'.'+file_ext)
                path1 = os.path.join(
                    lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[2])+'.'+file_ext)
                issame = True
            elif len(pair) == 4:
                path0 = os.path.join(
                    lfw_dir, pair[0], pair[0] + '_' + '%04d' % int(pair[1])+'.'+file_ext)
                path1 = os.path.join(
                    lfw_dir, pair[2], pair[2] + '_' + '%04d' % int(pair[3])+'.'+file_ext)
                issame = False
            # Only add the pair if both paths exist
            if os.path.exists(path0) and os.path.exists(path1):
                path_list.append((path0, path1, issame))
                issame_list.append(issame)
            else:
                nrof_skipped_pairs += 1
        # 记录路径不存在的 pairs 数量
        if nrof_skipped_pairs > 0:
            print('Skipped %d image pairs' % nrof_skipped_pairs)

        return path_list

    def __getitem__(self, index):
        (path_1, path_2, issame) = self.validation_images[index]
        image1, image2 = Image.open(path_1), Image.open(path_2)

        image1 = resize_image(
            # 要 resize 成的大小是一个列表
            image1, [self.image_size[1], self.image_size[0]], letterbox_image=True)
        image2 = resize_image(
            image2, [self.image_size[1], self.image_size[0]], letterbox_image=True)

        # preprocess_input() 用来将图像的像素值归一化到 [-1,1] 之间
        # np.transpose() 用来对数组进行转置, 可以用来交换数组的轴
        # 比如这里将通道放在了最前面, 为了遵循 B C H W, PIL.Image.open 打开的图像是 H W C 的形式
        image1, image2 = np.transpose(preprocess_input(np.array(image1, np.float32)), [
                                      2, 0, 1]), np.transpose(preprocess_input(np.array(image2, np.float32)), [2, 0, 1])

        # 返回的 numpy
        return image1, image2, issame

    # 返回测试集长度
    def __len__(self):
        return len(self.validation_images)


# 返回两张图片的tenosr就可以了, 不需要标签
class SCfaceDataset(datasets.ImageFolder):
    def __init__(self, dir, image_size, transform=None):
        super(SCfaceDataset, self).__init__(dir, transform)
        self.image_size = image_size
        self.dataset_path = dir
        persons = os.listdir(dir)
        persons.sort(key=int)
        self.pairs_list = []
        nrof_skipped_pairs = 0
        for i in range(len(persons)):
            query_path = os.path.join(
                self.dataset_path, persons[i], persons[i]+"_0002.jpg")
            origin_path = os.path.join(
                self.dataset_path, persons[i], persons[i]+"_0001.jpg")
            if os.path.exists(query_path) and os.path.exists(origin_path):
                self.pairs_list.append((query_path, origin_path))
            else:
                nrof_skipped_pairs += 1
        # 记录路径不存在的 pairs 数量
        if nrof_skipped_pairs > 0:
            print('Skipped %d image pairs' % nrof_skipped_pairs)
        self.length = len(self.pairs_list)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        (query_path, origin_path) = self.pairs_list[index]
        image1, image2 = Image.open(query_path), Image.open(origin_path)

        image1 = resize_image(
            # 要 resize 成的大小是一个列表
            image1, [self.image_size[1], self.image_size[0]], letterbox_image=True)
        image2 = resize_image(
            image2, [self.image_size[1], self.image_size[0]], letterbox_image=True)

        # preprocess_input() 用来将图像的像素值归一化到 [-1,1] 之间
        # np.transpose() 用来对数组进行转置, 可以用来交换数组的轴
        # 比如这里将通道放在了最前面, 为了遵循 B C H W, PIL.Image.open 打开的图像是 H W C 的形式
        image1, image2 = np.transpose(preprocess_input(np.array(image1, np.float32)), [
                                      2, 0, 1]), np.transpose(preprocess_input(np.array(image2, np.float32)), [2, 0, 1])
        return image1, image2
