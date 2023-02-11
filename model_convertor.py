import torch
import numpy as np
import os

from nets.arcface_ import Arcface
from utils.utils import get_num_classes

model_path = "/home/zk/project/arcface-pytorch/model_data/glint360k_iresnet50.pth"
backbone = "iresnet50"
idx = model_path.rfind('.')
model_save_path = model_path[:idx] + '_convert' + model_path[idx:]
annotation_path = "cls_train.txt"
num_classes = get_num_classes(annotation_path)

model = Arcface(num_classes=num_classes, backbone=backbone)
model1 = model.arcface
# model = iresnet50(dropout=0, num_features=512)
print('Load weights {}.'.format(model_path))

# ------------------------------------------------------#
#   根据预训练权重的Key和模型的Key进行加载
# ------------------------------------------------------#
model_dict = model1.state_dict()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
model1.load_state_dict(model_dict)
# ------------------------------------------------------#
#   显示没有匹配上的Key
# ------------------------------------------------------#
print("\nSuccessful Load Key:", str(load_key)[
    :500], "……\nSuccessful Load Key Num:", len(load_key))
print("\nFail To Load Key:", str(no_load_key)[
    :500], "……\nFail To Load Key num:", len(no_load_key))
print(
    "\n\033[1;33;44m温馨提示，head部分没有载入是正常现象，Backbone部分没有载入是错误的。\033[0m")
torch.save(model.state_dict(), model_save_path)
