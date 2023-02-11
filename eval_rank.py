import torch
import torch.backends.cudnn as cudnn
import numpy as np

from nets.arcface import Arcface
from utils.dataloader import SCfaceDataset


if __name__ == "__main__":
    # --------------------------------------#
    #   是否使用Cuda
    #   没有GPU可以设置成False
    # --------------------------------------#
    cuda = True
    # --------------------------------------#
    #   主干特征提取网络的选择
    #   mobilefacenet
    #   mobilenetv1
    #   iresnet18
    #   iresnet34
    #   iresnet50
    #   iresnet100
    #   iresnet200
    # --------------------------------------#
    # backbone = "mobilefacenet"
    backbone = "iresnet50"
    # --------------------------------------#
    #   输入图像大小
    # --------------------------------------#
    input_shape = [112, 112, 3]
    # --------------------------------------#
    #   训练好的权值文件
    # --------------------------------------#
    # model_path = "./logs/logs6/ep100-loss0.007-val_loss1.250.pth"
    # model_path = "./logs/logs6/ep096-loss0.007-val_loss1.227.pth"
    # model_path = "/home/zk/project/arcface-pytorch/logs/1_8_logs/ep003-acc0.733-val0.206.pth"
    # model_path = "./logs/logs7/ep100-loss0.008-val_loss13.744.pth"
    # model_path = "model_data/arcface_iresnet50.pth"
    # model_path = "model_data/arcface_mobilefacenet.pth"
    # model_path = "model_data/ms1mv3_iresnet50_convert.pth"
    model_path = "model_data/glint360k_iresnet50_convert.pth"
    # model_path = "model_data/ms1mv3_iresnet100_convert.pth"
    # model_path = "/home/zk/project/arcface-pytorch/logs/lfw2_origin/01_14_22_26/ep100-acc0.042-val0.117.pth"
    # --------------------------------------#
    #   LFW评估数据集的文件路径
    #   以及对应的txt文件
    # --------------------------------------#
    # lfw_dir_path = "datasets/lfw"
    # lfw_pairs_path = "model_data/lfw_pair.txt"
    # lfw_dir_path = "datasets/SCface/sc2_6"
    # lfw_pairs_path = "model_data/SCface_pair.txt"

    # --------------------------------------#
    #   评估的批次大小和记录间隔
    # --------------------------------------#
    batch_size = 1
    log_interval = 1
    # --------------------------------------#
    #   ROC图的保存路径
    # --------------------------------------#
    png_save_path = "model_data/roc_test.png"

    rank_dir_path = "datasets/SCface/sc2_6"
    # rank_dir_path = "datasets/lfw_SCface_test"
    # rank_dir_path = "datasets/lfw2_origin"

    RANK_loader = torch.utils.data.DataLoader(
        SCfaceDataset(dir=rank_dir_path, image_size=input_shape), batch_size=batch_size, shuffle=False)

    model = Arcface(backbone=backbone, mode="predict")
    # model_path = "/home/zk/project/arcface-pytorch/model_data/ms1mv3_iresnet50.pth"
    #
    # from nets.insightface import (iresnet18, iresnet34, iresnet50, iresnet100,
    #                               iresnet200)
    #
    # model = iresnet50(dropout=0.5, num_features=512)

    print('Loading weights into state dict...')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(
        model_path, map_location=device), strict=False)
    model = model.eval()

    if cuda:
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        model = model.cuda()

    with torch.no_grad():
        all_query, all_origin = None, None
        for _, (query, origin) in enumerate(RANK_loader):
            query, origin = query.type(
                torch.FloatTensor), origin.type(torch.FloatTensor)
            if cuda:
                query, origin = query.cuda(0), origin.cuda(0)
            out_a, out_p = model(query), model(origin)
            if all_query is None:
                all_query = out_a
                all_origin = out_p
            else:
                all_query = torch.cat((all_query, out_a))
                all_origin = torch.cat((all_origin, out_p))
        # print("all_query:", all_query.shape)
        # print("all_origin:", all_origin.shape)
        person_num = all_query.shape[0]
        ranks = np.zeros(10)
        print(ranks.shape)
        for index, query in enumerate(all_query):
            # shape: (128,1)
            query = query.view(-1, 1)
            # (120,128) X (128,1)
            score = torch.mm(all_origin, query)
            # 删除 score 中维度为 1 的那一维, (120)
            score = score.squeeze(1).cpu().numpy()
            print("score:", score)
            # from large to small
            origin_index = np.argsort(score)
            origin_index = origin_index[::-1]
            print("origin_index:", origin_index)
            rank = np.argwhere(origin_index == index)
            rank = rank.item()
            if rank < 10:
                ranks[rank:] += 1
        ranks /= person_num
        print('Rank1: %f Rank5: %f Rank10: %f ' %
              (ranks[0], ranks[4], ranks[9]))
