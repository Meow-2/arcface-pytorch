import os
from matplotlib import pyplot as plt
import scipy.signal


def get_acc(path):
    acc = []
    with open(path, 'r') as f:
        for line in f:
            acc.append(float(line))
    return acc


root_path = '/home/zk/project/arcface-pytorch/logs/lfw2_no_triplet/'
data_path = root_path
isFinetuning = root_path.find('Finetuning')
dataset_name = ""
if isFinetuning != -1:
    data_path = root_path[:isFinetuning]
    dataset_name = "Finetuning_"
data_path = data_path.rstrip('/')
dataset_name = dataset_name + data_path.rstrip('/').split('/')[-1]
# print(dataset_name)

path_list = os.listdir(root_path)
path_list = [x for x in path_list if x != "Finetuning" and x != "batch_1"]
acc_paths = []
for i in path_list:
    current_path = os.path.join(root_path, i)
    files = os.listdir(current_path)
    # subdir 就是current_path下的唯一一个子目录
    for j in files:
        current_file = os.path.join(current_path, j)
        if os.path.isdir(current_file):
            subdir_path = current_file
    acc_paths.append(os.path.join(subdir_path, "epoch_acc.txt"))

# acc_paths = []
# acc_paths.append("/home/zk/01_11_01_33/loss_2023_01_11_01_33_56/epoch_acc.txt")
# acc_paths.append(
#     "/home/zk/01_11_01_33/loss_2023_01_11_01_33_56/epoch_loss.txt")
# dataset_name = "CASIA-WebFace-LFW"
# path_list = ['acc', 'loss']
fig = plt.figure()

for acc_name, acc_path in zip(path_list, acc_paths):
    acc_list = get_acc(acc_path)
    epoch_num = len(acc_list)
    # acc_list = scipy.signal.savgol_filter(
    #     acc_list, 5 if epoch_num < 25 else 15, 3)
    X = range(epoch_num)
    plt.plot(X, acc_list,  linewidth=1, label=acc_name)

origin_acc = [0.65]*epoch_num
plt.plot(X, origin_acc,  linewidth=1, label="origin")

plt.grid(True)
plt.xlabel('Epoch')
plt.ylabel('Acc')
# plt.legend(loc="best")
fig.savefig(f"{dataset_name}_epoch_acc", dpi=fig.dpi)
