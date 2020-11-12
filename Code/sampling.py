#Сэмплирование позитивных и негативных примеров  
import torch
from torch_cluster import random_walk
from torch_geometric.utils import negative_sampling
from torch_geometric.utils import structured_negative_sampling
import numpy as np
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_sparse import SparseTensor

dataset =Planetoid(root='/tmp/Cora', name='Cora',transform=T.NormalizeFeatures())
data = dataset[0]
device = 'cpu'
x = data.x.to(device)
y = data.y.squeeze().to(device)

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
adj = SparseTensor(row=row, col=col, sparse_sizes=(140, 140)).to(device) #это edge_index для train

def pos_sample(batch):
        p=1
        q=1
        walk_length = 20
        walks_per_node = 10
        context_size = 10
        batch = batch.repeat(walks_per_node) #так как в лоадере мы не меняли размер батча, по дефолту он раве 1, а значит мы повторили 10 раз одну вершину 
        rowptr, col, _ = adj.csr()
        rowptr = rowptr.to(device)
        col = col.to(device)
        rw = random_walk(row, col, batch,  walk_length, p, q) #построили по нашим row,col. по одному рандом волку размером walk_length из каждой вершины в батче (в нашем случае это одна и та же вершина повторенная 10 раз)
        walks = []
        num_walks_per_rw = 1 + walk_length + 1 - context_size
        for j in range(num_walks_per_rw):
            walks.append(rw[:, j:j + context_size]) #теперь у нас внутри walks лежат 12 матриц размерам 10*10 
        return torch.cat(walks, dim=0) #Благодаря конкатенации теперь walks выглядит как матрица размером 120*10
    #в train_test мы проходим по загрузчику поэтапно, значит батч меняется от 1 до 140ой вершины, на каждую вершину мы получаем по матрице размером 120 на 10. 

    
row2 = ((row.tolist()))
col2 = ((col.tolist()))

a = []
a.append(row2)
a.append(col2)
a = torch.tensor(a)
def neg_sample(batch):
    walks_per_node = 10
    num_negative_samples=1
    batch = batch.repeat(walks_per_node * num_negative_samples) 
    neg_sam = structured_negative_sampling(a)
    b = []
    b.append(neg_sam[0].tolist())
    c =neg_sam[2].tolist()
    b.append(c)
    b = ((np.array(b).transpose()))
    b = torch.tensor(b).type(torch.LongTensor)
    return b # матрица размером 43*2
def sample(batch):
    if not isinstance(batch, torch.Tensor):
        batch = torch.tensor(batch).to(device)
    return pos_sample(batch),neg_sample(batch)

from torch.utils.data import DataLoader
from sampling import sample
loader = DataLoader(range(adj.sparse_size(0)),collate_fn=sample)
