
from torch.utils.data import DataLoader
from utils.dataloader import SCfaceDataset

dataset_path = '/home/zk/project/arcface-pytorch/datasets/SCface/sc2_6/'
image_size = [112, 112, 3]
dataset = SCfaceDataset(dir=dataset_path, image_size=image_size)
dataloader = DataLoader(SCfaceDataset(
    dir=dataset_path, image_size=image_size), batch_size=130, shuffle=False)
all_data = list(dataloader)
print(all_data[0].shape)
# print(image2.shape)
# for batch_i, (image_path1, image_path2) in enumerate(dataloader):
#   print(image_path1.shape)
#   print(image_path2.shape)
# print(len(dataset))
