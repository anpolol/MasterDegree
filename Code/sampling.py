#Сэмплирование позитивных и негативных примеров  
import torch
from torch_cluster import random_walk
device = 'cpu'

import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
dataset =Planetoid(root='/tmp/Cora', name='Cora',transform=T.NormalizeFeatures())
data = dataset[0]

x = data.x.to(device)
y = data.y.squeeze().to(device)

from torch_sparse import SparseTensor
row=[]
col =[]
x_new = [j for j in range(len(x))]
x_new = torch.IntTensor(x_new)
x_new = x_new[data.train_mask]
for j, i in enumerate(data.edge_index[0]):
    if i in x_new:
        if data.edge_index[1][j] in x_new:
                row.append(i)
                col.append(data.edge_index[1][j])
row = torch.tensor(row)
row = row.to(device)
col = torch.tensor(col)
col = col.to(device)

adj = SparseTensor(row=row, col=col, sparse_sizes=(140, 140)).to(device)

def pos_sample(batch):
        p=1
        q=1
        walk_length = 20
        walks_per_node = 10
        context_size = 10
        batch = batch.repeat(walks_per_node)
        rowptr, col, _ = adj.csr()
        rowptr = rowptr.to(device)
        col = col.to(device)
        rw = random_walk(row, col, batch,  walk_length, p, q)
        walks = []
        num_walks_per_rw = 1 + walk_length + 1 - context_size
        for j in range(num_walks_per_rw):
            walks.append(rw[:, j:j + context_size])
        return torch.cat(walks, dim=0)
def neg_sample(batch):
    walks_per_node = 10
    num_negative_samples = 1
    walk_length = 20 
    context_size = 10
    batch = batch.repeat(walks_per_node * num_negative_samples)

    rw = torch.randint(adj.sparse_size(0), (batch.size(0), walk_length)).to(device)
    rw = torch.cat([batch.view(-1, 1), rw], dim=-1)

    walks = []
    num_walks_per_rw = 1 + walk_length + 1 - context_size
    for j in range(num_walks_per_rw):
        walks.append(rw[:, j:j + context_size])
    return torch.cat(walks, dim=0)
def sample(batch):
    if not isinstance(batch, torch.Tensor):
        batch = torch.tensor(batch).to(device)
    return pos_sample(batch),neg_sample(batch)
