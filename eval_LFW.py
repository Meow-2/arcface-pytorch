import torch
import torch.backends.cudnn as cudnn

from nets.arcface import Arcface
from utils.dataloader import LFWDataset
from utils.utils_metrics import test


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
    # model_path = "model_data/arcface_mobilefacenet.pth"
    # model_path = "model_data/ms1mv3_iresnet50_convert.pth"
    model_path = "model_data/glint360k_iresnet50_convert.pth"
    # --------------------------------------#
    #   LFW评估数据集的文件路径
    #   以及对应的txt文件
    # --------------------------------------#
    # lfw_dir_path = "datasets/lfw"
    # lfw_pairs_path = "model_data/lfw_pair.txt"
    lfw_dir_path = "datasets/SCface/sc2_6"
    lfw_pairs_path = "model_data/SCface_pair.txt"
    # --------------------------------------#
    #   评估的批次大小和记录间隔
    # --------------------------------------#
    batch_size = 1
    log_interval = 1
    # --------------------------------------#
    #   ROC图的保存路径
    # --------------------------------------#
    png_save_path = "model_data/roc_test.png"

    test_loader = torch.utils.data.DataLoader(
        LFWDataset(dir=lfw_dir_path, pairs_path=lfw_pairs_path, image_size=input_shape), batch_size=batch_size, shuffle=False)

    model = Arcface(backbone=backbone, mode="predict")

    print('Loading weights into state dict...')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(
        model_path, map_location=device), strict=False)
    model = model.eval()

    if cuda:
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        model = model.cuda()

    test(test_loader, model, png_save_path, log_interval, batch_size, cuda)
