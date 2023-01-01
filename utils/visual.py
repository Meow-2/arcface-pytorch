from matplotlib import pyplot as plt
import scipy.signal


def get_epoch_acc(path):
    epoch_acc = []
    with open(path, 'r') as f:
        for line in f:
            epoch_acc.append(float(line))
    return epoch_acc


lfw_epoch_acc_path = '/home/zk/project/arcface-pytorch/logs/logs6/loss_2022_12_30_03_25_32/epoch_acc.txt'
lfw_origin_epoch_acc_path = '/home/zk/project/arcface-pytorch/logs/logs7/loss_2022_12_30_04_37_54/epoch_acc.txt'

lfw_epoch_acc = get_epoch_acc(lfw_epoch_acc_path)
lfw_origin_epoch_acc = get_epoch_acc(lfw_origin_epoch_acc_path)

epoch_num = len(lfw_epoch_acc)
X = range(epoch_num)
origin_acc = [0.65]*epoch_num

fig = plt.figure()

# 曲线平滑
lfw_epoch_acc = scipy.signal.savgol_filter(
    lfw_epoch_acc, 5 if epoch_num < 25 else 15, 3)
lfw_origin_epoch_acc = scipy.signal.savgol_filter(
    lfw_origin_epoch_acc, 5 if epoch_num < 25 else 15, 3)
plt.plot(X, lfw_epoch_acc, 'red', linewidth=1,
         label='fine tuning with lfw^')
plt.plot(X, lfw_origin_epoch_acc, 'green',
         linewidth=1, label='fine tuning with lfw')
plt.plot(X, origin_acc, 'blue', linewidth=1, label='no fine tuning')
plt.grid(True)
plt.xlabel('Epoch')
plt.ylabel('Acc')
plt.legend(loc="upper right")
fig.savefig('epoch_acc', dpi=fig.dpi)
