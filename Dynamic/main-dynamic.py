import argparse
import torch
import torch.nn.functional as F
from networks import GNNs
from dynamicGnn import DynamicGNN
from torch import tensor
from torch.optim import Adam
import numpy as np
import os
import random
from torch_geometric.data import Data
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from torch_geometric.nn import GCNConv, GINConv, SAGEConv, GraphConv, TransformerConv, ResGatedGraphConv, ChebConv, GATConv, SGConv, GeneralConv
from torch_geometric.loader import DataLoader
import os.path as osp
from NeuroGraph.utils import augment_graph, info_nce_loss
import sys
import time
from torch_geometric.data import Batch
from sklearn.decomposition import PCA

# 参数设置
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='DynHCPGender')
parser.add_argument('--runs', type=int, default=1)
parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
parser.add_argument('--seed', type=int, default=123)
parser.add_argument('--model', type=str, default="TransformerConv")
parser.add_argument('--hidden1', type=int, default=128)
parser.add_argument('--hidden2', type=int, default=32)
parser.add_argument('--num_heads', type=int, default=1)
parser.add_argument('--num_layers', type=int, default=3)
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--early_stopping', type=int, default=50)
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--weight_decay', type=float, default=0.0005)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--contrastive_weight', type=float, default=0.1)
args = parser.parse_args()

# 设置随机种子
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(args.seed)

# 路径设置
path = "base_params/"
res_path = "base_results/"
path_data = "../../data/"
os.makedirs(path, exist_ok=True)
os.makedirs(res_path, exist_ok=True)

# 日志函数
def logger(info):
    with open(os.path.join(res_path, 'dynamic_results.csv'), 'a') as f:
        print(info, file=f)

log = "dataset,model,hidden,num_layers,epochs,batch_size,loss,acc,std"
logger(log)

# 数据加载
def load_data(dataset_name):
    dataset_path = os.path.join(path_data, dataset_name, "processed", f"{dataset_name}.pt")
    if dataset_name == 'DynHCPGender':
        dataset_raw = torch.load(dataset_path)
        dataset, labels = [], []
        for v in dataset_raw:
            batches = v.get('batches')
            if len(batches) > 0:
                for b in batches:
                    y = b.y[0].item()
                    dataset.append(b)
                    labels.append(y)
    else:
        dataset = torch.load(dataset_path)
        labels = dataset['labels']
        dataset = dataset['batches']
    return dataset, labels

dataset, labels = load_data(args.dataset)
print(f"Dataset {args.dataset} loaded successfully!")

# 数据集分割
train_tmp, test_indices = train_test_split(list(range(len(labels))), test_size=0.2, stratify=labels, random_state=args.seed)
tmp = [dataset[i] for i in train_tmp]
labels_tmp = [labels[i] for i in train_tmp]
train_indices, val_indices = train_test_split(list(range(len(labels_tmp))), test_size=0.125, stratify=labels_tmp, random_state=args.seed)

train_dataset = [tmp[i] for i in train_indices]
val_dataset = [tmp[i] for i in val_indices]
test_dataset = [dataset[i] for i in test_indices]

print(f"Dataset {args.dataset} split into train {len(train_dataset)}, val {len(val_dataset)}, test {len(test_dataset)}")

args.num_features, args.num_classes = 100, len(np.unique(labels))
print(f"Number of features: {args.num_features}, Number of classes: {args.num_classes}")

# 模型、损失函数和优化器
criterion = torch.nn.CrossEntropyLoss()

def train(model, optimizer, train_loader):
    model.train()
    total_loss = 0
    for data in train_loader:
        data = data.to(args.device)
        aug_data = augment_graph(data).to(args.device)
        
        optimizer.zero_grad()
        out, proj = model(data)
        aug_out, aug_proj = model(aug_data)
        
        task_loss = criterion(out.reshape(1, -1), data[0].y)
        contrastive_loss = info_nce_loss(proj, data.batch) + info_nce_loss(aug_proj, data.batch)
        
        loss = task_loss + args.contrastive_weight * contrastive_loss
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    return total_loss / len(train_loader)

def test(model, loader):
    model.eval()
    correct = 0
    for data in loader:
        data = data.to(args.device)
        with torch.no_grad():
            out, _ = model(data)
            pred = out.argmax(dim=1)
            correct += int((pred == data[0].y).sum())
    return correct / len(loader)

# 主训练循环
def main():
    for run in range(args.runs):
        set_seed(args.seed + run)
        
        gnn = eval(args.model)
        model = DynamicGNN(args.num_features, args.hidden1, args.hidden2, args.num_heads, 
                           args.num_layers, gnn, args.dropout, args.num_classes).to(args.device)
        optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        
        best_val_acc = 0
        patience = 0
        for epoch in range(args.epochs):
            loss = train(model, optimizer, train_dataset)
            val_acc = test(model, val_dataset)
            test_acc = test(model, test_dataset)
            
            if epoch % 10 == 0:
                print(f"Epoch: {epoch:03d}, Loss: {loss:.4f}, Val Acc: {val_acc:.4f}, Test Acc: {test_acc:.4f}")
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience = 0
                torch.save(model.state_dict(), f"{path}{args.dataset}_{args.model}_best.pt")
            else:
                patience += 1
            
            if patience >= args.early_stopping:
                print("Early stopping!")
                break
        
        # 最终测试
        model.load_state_dict(torch.load(f"{path}{args.dataset}_{args.model}_best.pt"))
        final_test_acc = test(model, test_dataset)
        print(f"Final Test Accuracy: {final_test_acc:.4f}")
        
        # 记录结果
        log_info = f"{args.dataset},{args.model},{args.hidden1},{args.num_layers},{args.epochs},{args.batch_size},{loss:.4f},{final_test_acc:.4f}"
        logger(log_info)

if __name__ == "__main__":
    start_time = time.time()
    main()
    print(f"Total time: {time.time() - start_time:.2f} seconds")