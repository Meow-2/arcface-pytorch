import os

import argparse
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim
from torch.utils.data import DataLoader

from nets.arcface import Arcface
from nets.arcface_training import get_lr_scheduler, set_optimizer_lr
from utils.callback import LossHistory
from utils.dataloader import FacenetDataset, LFWDataset, SCfaceDataset, dataset_collate
from utils.utils import get_num_classes, show_config
from utils.utils_fit import fit_one_epoch
from utils.utils_txt import txt_annotation

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset')
    parser.add_argument('--epoch')
    parser.add_argument('--model', type=str, default="")
    args = parser.parse_args()

    from datetime import datetime
    date = datetime.now()
    month = datetime.strftime(date, '%m')
    day = datetime.strftime(date, '%d')
    hour = datetime.strftime(date, '%H')
    minute = datetime.strftime(date, '%M')

    seed = 4396
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    # -------------------------------#
    #   是否使用Cuda
    #   没有GPU可以设置成False
    # -------------------------------#
    Cuda = True
    # ---------------------------------------------------------------------#
    #   distributed     用于指定是否使用单机多卡分布式运行
    #                   终端指令仅支持Ubuntu。CUDA_VISIBLE_DEVICES用于在Ubuntu下指定显卡。
    #                   Windows系统下默认使用DP模式调用所有显卡，不支持DDP。
    #   DP模式：
    #       设置            distributed = False
    #       在终端中输入    CUDA_VISIBLE_DEVICES=0,1 python train.py
    #   DDP模式：
    #       设置            distributed = True
    #       在终端中输入    CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train.py
    # ---------------------------------------------------------------------#
    distributed = False
    # ---------------------------------------------------------------------#
    #   sync_bn     是否使用sync_bn，DDP模式多卡可用
    # ---------------------------------------------------------------------#
    sync_bn = False
    # ---------------------------------------------------------------------#
    #   fp16        是否使用混合精度训练
    #               可减少约一半的显存、需要pytorch1.7.1以上
    # ---------------------------------------------------------------------#
    fp16 = False
    # --------------------------------------------------------#
    #   指向根目录下的cls_train.txt，读取人脸路径与标签
    # --------------------------------------------------------#
    annotation_path = "cls_train.txt"
    # --------------------------------------------------------#
    #   输入图像大小
    # --------------------------------------------------------#
    input_shape = [112, 112, 3]
    # --------------------------------------------------------#
    #   主干特征提取网络的选择
    #   mobilefacenet
    #   mobilenetv1
    #   iresnet18
    #   iresnet34
    #   iresnet50
    #   iresnet100
    #   iresnet200
    #
    #   除了mobilenetv1外，其它的backbone均可从0开始训练。
    #   这是由于mobilenetv1没有残差边，收敛速度慢，因此建议：
    #   如果使用mobilenetv1为主干,  则设置pretrain = True
    #   如果使用其它网络为主干，    则设置pretrain = False
    # --------------------------------------------------------#
    backbone = "mobilefacenet"
    # ----------------------------------------------------------------------------------------------------------------------------#
    #   如果训练过程中存在中断训练的操作，可以将model_path设置成logs文件夹下的权值文件，将已经训练了一部分的权值再次载入。
    #   同时修改下方的训练的参数，来保证模型epoch的连续性。
    #
    #   当model_path = ''的时候不加载整个模型的权值。
    #
    #   此处使用的是整个模型的权重，因此是在train.py进行加载的，pretrain不影响此处的权值加载。
    #   如果想要让模型从主干的预训练权值开始训练，则设置model_path = ''，pretrain = True，此时仅加载主干。
    #   如果想要让模型从0开始训练，则设置model_path = ''，pretrain = Fasle，此时从0开始训练。
    # ----------------------------------------------------------------------------------------------------------------------------#
    # model_path = "/home/zk/project/arcface-pytorch/model_data/arcface_mobilefacenet.pth"
    # model_path = ''
    model_path = args.model
    # ----------------------------------------------------------------------------------------------------------------------------#
    #   是否使用主干网络的预训练权重，此处使用的是主干的权重，因此是在模型构建的时候进行加载的。
    #   如果设置了model_path，则主干的权值无需加载，pretrained的值无意义。
    #   如果不设置model_path，pretrained = True，此时仅加载主干开始训练。
    #   如果不设置model_path，pretrained = False，此时从0开始训练。
    #   除了mobilenetv1外，其它的backbone均未提供预训练权重。
    # ----------------------------------------------------------------------------------------------------------------------------#
    pretrained = False

    # ----------------------------------------------------------------------------------------------------------------------------#
    #   显存不足与数据集大小无关，提示显存不足请调小batch_size。
    #   受到BatchNorm层影响，不能为1。
    #
    #   在此提供若干参数设置建议，各位训练者根据自己的需求进行灵活调整：
    #   （一）从预训练权重开始训练：
    #       Adam：
    #           Init_Epoch = 0，Epoch = 100，optimizer_type = 'adam'，Init_lr = 1e-3，weight_decay = 0。
    #       SGD：
    #           Init_Epoch = 0，Epoch = 100，optimizer_type = 'sgd'，Init_lr = 1e-2，weight_decay = 5e-4。
    #       其中：UnFreeze_Epoch可以在100-300之间调整。
    #   （二）batch_size的设置：
    #       在显卡能够接受的范围内，以大为好。显存不足与数据集大小无关，提示显存不足（OOM或者CUDA out of memory）请调小batch_size。
    #       受到BatchNorm层影响，batch_size最小为2，不能为1。
    #       正常情况下Freeze_batch_size建议为Unfreeze_batch_size的1-2倍。不建议设置的差距过大，因为关系到学习率的自动调整。
    # ----------------------------------------------------------------------------------------------------------------------------#
    # ------------------------------------------------------#
    #   训练参数
    #   Init_Epoch      模型当前开始的训练世代
    #   Epoch           模型总共训练的epoch
    #   batch_size      每次输入的图片数量
    # ------------------------------------------------------#
    Init_Epoch = 0
    Epoch = int(args.epoch)
    batch_size = 64

    # ------------------------------------------------------------------#
    #   其它训练参数：学习率、优化器、学习率下降有关
    # ------------------------------------------------------------------#
    # ------------------------------------------------------------------#
    #   Init_lr         模型的最大学习率
    #   Min_lr          模型的最小学习率，默认为最大学习率的0.01
    # ------------------------------------------------------------------#
    Init_lr = 1e-2
    Min_lr = Init_lr * 0.01
    # ------------------------------------------------------------------#
    #   optimizer_type  使用到的优化器种类，可选的有adam、sgd
    #                   当使用Adam优化器时建议设置  Init_lr=1e-3
    #                   当使用SGD优化器时建议设置   Init_lr=1e-2
    #   momentum        优化器内部使用到的momentum参数
    #   weight_decay    权值衰减，可防止过拟合
    #                   adam会导致weight_decay错误，使用adam时建议设置为0。
    # ------------------------------------------------------------------#
    # 学习器的参数
    optimizer_type = "sgd"
    momentum = 0.9
    weight_decay = 5e-4
    # ------------------------------------------------------------------#
    #   lr_decay_type   使用到的学习率下降方式，可选的有step、cos
    # ------------------------------------------------------------------#
    lr_decay_type = "cos"
    # ------------------------------------------------------------------#
    #   save_period     多少个epoch保存一次权值，默认每个世代都保存
    # ------------------------------------------------------------------#
    save_period = 1
    # ------------------------------------------------------------------#
    #   save_dir        权值与日志文件保存的文件夹
    # ------------------------------------------------------------------#
    # dataset = 'lfw_no_triplet_1'
    dataset = args.dataset
    txt_annotation(dataset)
    save_dir = 'logs/'+dataset+'/' + month+'_' + day + \
        '_' + hour + '_' + minute
    # save_dir = 'logs/1_8_logs_with_triplet_7'
    # ------------------------------------------------------------------#
    #   用于设置是否使用多线程读取数据
    #   开启后会加快数据读取速度，但是会占用更多内存
    #   内存较小的电脑可以设置为2或者0
    # ------------------------------------------------------------------#
    num_workers = 4
    # ------------------------------------------------------------------#
    #   是否开启LFW评估
    # ------------------------------------------------------------------#
    eval_rank = True
    lfw_eval_flag = False
    # rank_dir_path = "datasets/SCface/sc2_6"
    rank_dir_path = "datasets/lfw_SCface_test"
    # ------------------------------------------------------------------#
    #   LFW评估数据集的文件路径和对应的txt文件
    # ------------------------------------------------------------------#
    # lfw_dir_path = "datasets/SCface/sc2_6"
    # lfw_pairs_path = "model_data/SCface_pair.txt"

    # lfw_dir_path = "datasets/lfw_96_112"
    # lfw_pairs_path = "model_data/lfw_pair.txt"
    lfw_dir_path = "datasets/lfw_SCface_test"
    lfw_pairs_path = "datasets/lfw_SCface_test_pair.txt"
    # ------------------------------------------------------#
    #   设置用到的显卡
    # ------------------------------------------------------#
    ngpus_per_node = torch.cuda.device_count()
    if distributed:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        device = torch.device("cuda", local_rank)
        if local_rank == 0:
            print(
                f"[{os.getpid()}] (rank = {rank}, local_rank = {local_rank}) training...")
            print("Gpu Device Count : ", ngpus_per_node)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        local_rank = 0
        rank = 0

    # 获取种类数量
    num_classes = get_num_classes(annotation_path)
    # ---------------------------------#
    #   载入模型并加载预训练权重
    # ---------------------------------#
    model = Arcface(num_classes=num_classes,
                    backbone=backbone, pretrained=pretrained)

    if model_path != '':
        # ------------------------------------------------------#
        #   权值文件请看README，百度网盘下载
        # ------------------------------------------------------#
        if local_rank == 0:
            print('Load weights {}.'.format(model_path))

        # ------------------------------------------------------#
        #   根据预训练权重的Key和模型的Key进行加载
        # ------------------------------------------------------#
        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location=device)
        # 先把所有能载入的key都放进 temp_dict 里
        load_key, no_load_key, temp_dict = [], [], {}
        for k, v in pretrained_dict.items():
            if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                temp_dict[k] = v
                load_key.append(k)
            else:
                no_load_key.append(k)
        model_dict.update(temp_dict)  # 如果 key 已经存在, 則会被更新为新的值
        model.load_state_dict(model_dict)
        # ------------------------------------------------------#
        #   显示没有匹配上的Key
        # ------------------------------------------------------#
        if local_rank == 0:
            print("\nSuccessful Load Key:", str(load_key)[
                  :500], "……\nSuccessful Load Key Num:", len(load_key))
            print("\nFail To Load Key:", str(no_load_key)[
                  :500], "……\nFail To Load Key num:", len(no_load_key))
            print(
                "\n\033[1;33;44m温馨提示，head部分没有载入是正常现象，Backbone部分没有载入是错误的。\033[0m")

    # ----------------------#
    #   记录Loss
    # ----------------------#
    if local_rank == 0:
        loss_history = LossHistory(save_dir, model, input_shape=input_shape)
    else:
        loss_history = None

    # ------------------------------------------------------------------#
    #   torch 1.2不支持amp，建议使用torch 1.7.1及以上正确使用fp16
    #   因此torch1.2这里显示"could not be resolve"
    # ------------------------------------------------------------------#
    # 这里不知道是干嘛的
    if fp16:
        from torch.cuda.amp import GradScaler as GradScaler
        scaler = GradScaler()
    else:
        scaler = None

    model_train = model.train()  # return self
    # ----------------------------#
    #   多卡同步Bn
    # ----------------------------#
    if sync_bn and ngpus_per_node > 1 and distributed:
        model_train = torch.nn.SyncBatchNorm.convert_sync_batchnorm(
            model_train)
    elif sync_bn:
        print("Sync_bn is not support in one gpu or not distributed.")

    if Cuda:
        if distributed:
            # ----------------------------#
            #   多卡平行运行
            # ----------------------------#
            model_train = model_train.cuda(local_rank)
            model_train = torch.nn.parallel.DistributedDataParallel(
                model_train, device_ids=[local_rank], find_unused_parameters=True)
        else:
            model_train = torch.nn.DataParallel(model)
            cudnn.benchmark = True  # 如果输入的尺寸一致, 可以用来加速
            model_train = model_train.cuda()  # 把模型放到 gpu 上

    # ---------------------------------#
    #   LFW估计
    # ---------------------------------#
    # def __getitem__():
    #   return image1, image2, issame
    # drop_last = False
    LFW_loader = torch.utils.data.DataLoader(
        LFWDataset(dir=lfw_dir_path, pairs_path=lfw_pairs_path, image_size=input_shape), batch_size=32, shuffle=False) if lfw_eval_flag else None
    RANK_loader = torch.utils.data.DataLoader(
        SCfaceDataset(dir=rank_dir_path, image_size=input_shape), batch_size=130, shuffle=False)

    # -------------------------------------------------------#
    #   0.01用于验证，0.99用于训练
    # -------------------------------------------------------#
    # 可以考虑设成 0
    # val_split = 0.01
    # 读入 cls_train.txt
    with open(annotation_path, "r") as f:
        lines = f.readlines()
    # 随机打乱
    # np.random.seed(10101)
    # np.random.shuffle(lines)
    # np.random.seed(None)

    # num_val = int(len(lines)*val_split)
    # num_train = len(lines) - num_val
    num_train = len(lines)

    # 打印所有的键值对
    # show_config(
    #     num_classes=num_classes, backbone=backbone, model_path=model_path, input_shape=input_shape,
    #     Init_Epoch=Init_Epoch, Epoch=Epoch, batch_size=batch_size,
    #     Init_lr=Init_lr, Min_lr=Min_lr, optimizer_type=optimizer_type, momentum=momentum, lr_decay_type=lr_decay_type,
    #     save_period=save_period, save_dir=save_dir, num_workers=num_workers, num_train=num_train, num_val=num_val
    # )
    show_config(
        num_classes=num_classes, backbone=backbone, input_shape=input_shape,
        Init_Epoch=Init_Epoch, Epoch=Epoch, batch_size=batch_size,
        Init_lr=Init_lr, Min_lr=Min_lr, optimizer_type=optimizer_type, momentum=momentum, lr_decay_type=lr_decay_type,
        save_period=save_period, num_workers=num_workers, num_train=num_train
    )

    if True:
        # -------------------------------------------------------------------#
        #   判断当前batch_size，自适应调整学习率
        # -------------------------------------------------------------------#
        nbs = 64
        lr_limit_max = 1e-3 if optimizer_type == 'adam' else 1e-1  # 1e-1
        lr_limit_min = 3e-4 if optimizer_type == 'adam' else 5e-4  # 5e-4
        # Init_lr = 1e-2
        # Min_lr = Init_lr * 0.01
        # 根据 batch_size, 调整学习率
        Init_lr_fit = min(max(batch_size / nbs * Init_lr,
                          lr_limit_min), lr_limit_max)
        Min_lr_fit = min(max(batch_size / nbs * Min_lr,
                         lr_limit_min * 1e-2), lr_limit_max * 1e-2)

        # ---------------------------------------#
        #   根据optimizer_type选择优化器
        # ---------------------------------------#
        optimizer = {
            'adam': optim.Adam(model.parameters(), Init_lr_fit, betas=(momentum, 0.999), weight_decay=weight_decay),
            'sgd': optim.SGD(model.parameters(), Init_lr_fit, momentum=momentum, nesterov=True, weight_decay=weight_decay)
        }[optimizer_type]

        # ---------------------------------------#
        #   获得学习率下降的公式
        # ---------------------------------------#
        lr_scheduler_func = get_lr_scheduler(
            lr_decay_type, Init_lr_fit, Min_lr_fit, Epoch)

        # ---------------------------------------#
        #   判断每一个世代的长度
        # ---------------------------------------#
        epoch_step = num_train // batch_size  # 整数除法, 返回商的整数部分
        # epoch_step_val = num_val // batch_size  # 整数除法, 返回商的整数部分

        # 至少有一个 batch_size 张图片
        # if epoch_step == 0 or epoch_step_val == 0:
        #     raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")
        if epoch_step == 0:
            raise ValueError("数据集过小，无法继续进行训练，请扩充数据集。")

        # ---------------------------------------#
        #   构建数据集加载器。
        # ---------------------------------------#
        # num_val = int(len(lines)*val_split)
        # num_train = len(lines) - num_val
        # lines[:num_train] :num_train 是左闭右开区间
        train_dataset = FacenetDataset(
            input_shape, lines[:num_train], random=False)
        # val_dataset = FacenetDataset(
        #     input_shape, lines[num_train:], random=False)

        if distributed:
            train_sampler = torch.utils.data.distributed.DistributedSampler(
                train_dataset, shuffle=True,)
            # val_sampler = torch.utils.data.distributed.DistributedSampler(
            #     val_dataset, shuffle=False,)
            batch_size = batch_size // ngpus_per_node
            shuffle = False
        else:
            train_sampler = None
            val_sampler = None
            shuffle = True

# dataset (Dataset): 要加载的数据集
# batch_size (int, optional): 每个batch中的样本数(默认: 1)。
# shuffle (bool, optional): 每个epoch是否打乱样本顺序(默认: False)。
# sampler (Sampler, optional): 这个参数可以在训练时指定对样本抽样的策略。这可以用于不打乱数据的情况下，让每一个epoch使用一个新的子集。
# batch_sampler (Sampler, optional): 与batch_size和shuffle参数互斥。可以手动指定每个batch中样本的索引。
# num_workers (int, optional): 读取数据的进程数(默认: 0)。
# collate_fn (callable, optional): 合并batch的函数，默认为default_collate。
# pin_memory (bool, optional): 如果为True，将会在返回之前将数据复制到CUDA固定内存中(默认: False)。
# drop_last (bool, optional): 如果为True，则当样本数不能被batch_size整除时，丢弃最后一个不完整的batch(默认: False)。

        # sampler = None
        gen = DataLoader(train_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
                         drop_last=True, collate_fn=dataset_collate, sampler=train_sampler)
        # gen_val = DataLoader(val_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers, pin_memory=True,
        #                      drop_last=True, collate_fn=dataset_collate, sampler=val_sampler)

        for epoch in range(Init_Epoch, Epoch):
            if distributed:
                train_sampler.set_epoch(epoch)

            # 根据当前的 epoch 设置学习率
            set_optimizer_lr(optimizer, lr_scheduler_func, epoch)

            # 跑一个 epoch
            fit_one_epoch(model_train, model, loss_history, optimizer, epoch, epoch_step,  gen,
                          Epoch, Cuda, LFW_loader, RANK_loader, lfw_eval_flag, eval_rank, fp16, scaler, save_period, save_dir, local_rank)

        if local_rank == 0:
            loss_history.writer.close()
