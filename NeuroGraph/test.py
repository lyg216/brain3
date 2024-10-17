# from NeuroGraph import utils
# import numpy as np
# from nilearn.image import load_img



# img = load_img("NeuroGraph/data/raw/1.nii.gz")
# regs = np.loadtxt("NeuroGraph/data/raw/1.txt")
# fmri = img.get_fdata()
# fc = utils.preprocess(fmri, regs,100)
# # fc = np.load("NeuroGraph/data/fc.npy")
# print(fc.shape) 
# data = utils.construct_data(fc, 1)
# print(data)
from NeuroGraph import utils
import numpy as np
from nilearn.image import load_img
import torch
from torch_geometric.data import Data, Batch
from dynamicGnn import DynamicGNN

# 加载和预处理数据
img = load_img("NeuroGraph/data/raw/1.nii.gz")
regs = np.loadtxt("NeuroGraph/data/raw/1.txt")
fmri = img.get_fdata()
fc = utils.preprocess(fmri, regs, 100)
print("Functional connectivity shape:", fc.shape)

# 构建图数据
data = utils.construct_data(fc, 1)
print("Graph data:", data)

# 创建一个批次
batch = Batch.from_data_list([data])

# 初始化模型
model = DynamicGNN(num_features=100, hidden1=128, hidden2=64, num_heads=4, 
                   num_layers=3, gnn=torch.nn.Module, dropout=0.5, num_classes=2)

# 前向传播
out, proj = model(batch)
print("Model output shape:", out.shape)
print("Projection output shape:", proj.shape)

# 数据增强示例
aug_data = utils.augment_graph(data)
print("Augmented graph data:", aug_data)

# 对比损失计算示例
features = torch.randn(10, 64)  # 假设有10个图,每个图的特征维度是64
batch = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])  # 5个图,每个图2个视图
loss = utils.info_nce_loss(features, batch)
print("Contrastive loss:", loss.item())