import os

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from .utils import get_lr
from .utils_metrics import evaluate


# gen 是训练集的 Dataloader
# gen_val 是验证集的 Dataloader
# model_train 和 model 应该是没有区别的
# loss_history = LossHistory(save_dir, model, input_shape=input_shape) , gpu 0 用来记录 loss 的函数
# optimizer 优化器
# epoch 当前 epoch
# epoch_step 为每个 epoch 的训练集 batch 数量
# epoch_step_val 为每个 epoch 的验证集 batch 数量
# Epoch 为 Epoch 总数
# cuda 为 bool 类型, 是否使用 CUDA
# 测试数据集的 Dataloader
# 是否要启用测试数据集 lfw_eval_flag
# fp16 = False, scaler = None, local_rank=0
# save_period 每隔多少个 epoch 保存一次模型
# save_dir log保存的文件夹
def fit_one_epoch(model_train, model, loss_history, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, Epoch, cuda, test_loader, lfw_eval_flag, fp16, scaler, save_period, save_dir, local_rank=0):
    total_loss = 0
    total_accuracy = 0

    val_total_loss = 0
    val_total_accuracy = 0

    if local_rank == 0:
        print('Start Train')
        pbar = tqdm(
            total=epoch_step, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3)
    model_train.train()
    # 对每一个 batch 进行操作
    for iteration, batch in enumerate(gen):
        if iteration >= epoch_step:
            break
        # 取数据 (图片, 标签), 放进 cuda 里
        images, labels = batch
        with torch.no_grad():
            if cuda:
                images = images.cuda(local_rank)
                labels = labels.cuda(local_rank)

        # ----------------------#
        #   清零梯度
        # ----------------------#
        optimizer.zero_grad()
        if not fp16:
            #
            outputs = model_train(images, labels, mode="train")
            # -1 表示最后一维, 对最后一维进行归一化
            # nn.NLLLoss() 接受两个参数: 一个预测概率分布和一个真实的标签
            loss = nn.NLLLoss()(F.log_softmax(outputs, -1), labels)

            loss.backward()
            optimizer.step()
        else:
            from torch.cuda.amp import autocast
            with autocast():
                outputs = model_train(images, labels, mode="train")
                loss = nn.NLLLoss()(F.log_softmax(outputs, -1), labels)
            # ----------------------#
            #   反向传播
            # ----------------------#
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        with torch.no_grad():
            # torch.argmax() 返回的是最大值的索引, 如果该索引等于 labels, 则说明预测正确了
            # torch.mean() 会先将 bool 类型的矩阵转化为浮点数类型, 再来计算均值, 也就是准确率
            accuracy = torch.mean((torch.argmax(
                F.softmax(outputs, dim=-1), dim=-1) == labels).type(torch.FloatTensor))

        total_loss += loss.item()
        total_accuracy += accuracy.item()

        if local_rank == 0:
            # iteration 是index, 所以需要 +1
            pbar.set_postfix(**{'total_loss': total_loss / (iteration + 1),
                                'accuracy': total_accuracy / (iteration + 1),
                                'lr': get_lr(optimizer)})
            pbar.update(1)

    # 这个在验证集上进行验证, 没有什么用
    if local_rank == 0:
        pbar.close()
        print('Finish Train')
        print('Start Validation')
        pbar = tqdm(total=epoch_step_val,
                    desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3)
    model_train.eval()
    for iteration, batch in enumerate(gen_val):
        if iteration >= epoch_step_val:
            break
        images, labels = batch
        with torch.no_grad():
            if cuda:
                images = images.cuda(local_rank)
                labels = labels.cuda(local_rank)

            optimizer.zero_grad()
            outputs = model_train(images, labels, mode="train")
            loss = nn.NLLLoss()(F.log_softmax(outputs, -1), labels)

            accuracy = torch.mean((torch.argmax(
                F.softmax(outputs, dim=-1), dim=-1) == labels).type(torch.FloatTensor))

            val_total_loss += loss.item()
            val_total_accuracy += accuracy.item()

        if local_rank == 0:
            pbar.set_postfix(**{'total_loss': val_total_loss / (iteration + 1),
                                'accuracy': val_total_accuracy / (iteration + 1),
                                'lr': get_lr(optimizer)})
            pbar.update(1)

    # 这个很有用
    if lfw_eval_flag:
        print("开始进行LFW数据集的验证。")
        labels, distances = [], []
        for _, (data_a, data_p, label) in enumerate(test_loader):
            with torch.no_grad():
                data_a, data_p = data_a.type(
                    torch.FloatTensor), data_p.type(torch.FloatTensor)
                if cuda:
                    data_a, data_p = data_a.cuda(
                        local_rank), data_p.cuda(local_rank)

                out_a, out_p = model_train(data_a), model_train(data_p)
                dists = torch.sqrt(torch.sum((out_a - out_p) ** 2, 1))
            distances.append(dists.data.cpu().numpy())
            labels.append(label.data.cpu().numpy())

        labels = np.array([sublabel for label in labels for sublabel in label])
        distances = np.array(
            [subdist for dist in distances for subdist in dist])
        _, _, accuracy, val, val_std, far, best_thresholds = evaluate(
            distances, labels)

    if local_rank == 0:
        pbar.close()
        print('Finish Validation')

        # 输出 LFW 验证信息
        if lfw_eval_flag:
            # print('Accuracy: %2.5f+-%2.5f' %
            #       (np.mean(accuracy), np.std(accuracy)))
            print('Accuracy: %2.5f+-%2.5f' %
                  (np.mean(accuracy), np.std(accuracy)))  # 均值, 方差
            print('Best_thresholds: %2.5f' % best_thresholds)
            print('Validation rate: %2.5f+-%2.5f @ FAR=%2.5f' %
                  (val, val_std, far))  # 均值, 方差, FAR

        # loss_history.append_loss(epoch, np.mean(accuracy) if lfw_eval_flag else total_accuracy /
        #                          epoch_step, total_loss / epoch_step, val_total_loss / epoch_step_val)
        # print('Total Loss: %.4f' % (total_loss / epoch_step))
        # if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
        #     torch.save(model.state_dict(), os.path.join(save_dir, 'ep%03d-loss%.3f-val_loss%.3f.pth' %
        #                ((epoch+1), total_loss / epoch_step, val_total_loss / epoch_step_val)))
        loss_history.append_loss(epoch, np.mean(accuracy) if lfw_eval_flag else total_accuracy /
                                 epoch_step, total_loss / epoch_step, val)
        # print('Total Loss: %.4f' % (total_loss / epoch_step))
        if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
            torch.save(model.state_dict(), os.path.join(save_dir, 'ep%03d-acc%.3f-val%.3f.pth' %
                       ((epoch+1), np.mean(accuracy), val)))
