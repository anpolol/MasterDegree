import torch
from torch_cluster import random_walk
from torch_geometric.utils import negative_sampling
from torch_geometric.utils import structured_negative_sampling,batched_negative_sampling
import numpy as np
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_sparse import SparseTensor
from torch.utils.data import DataLoader
from torch_geometric.utils.num_nodes import maybe_num_nodes
try:
    import torch_cluster  # noqa
    random_walk = torch.ops.torch_cluster.random_walk
except ImportError:
    random_walk = None
    
class RWSampler():
    def __init__(self, data,device,walk_length=1,p=1,q=1,walks_per_node=1, **kwargs):
        self.data = data
        self.device = 'cpu'
        self.mask = data.train_mask
        self.p=p
        self.q=q
        self.walk_length = walk_length
        row=[]
        col =[]
        self.x_new = [j for j in range(len(self.data.x))]
        self.x_new = torch.IntTensor(self.x_new)
        self.x_new = self.x_new[self.mask]
        for j, i in enumerate(self.data.edge_index[0]):
            if i in self.x_new:
                if self.data.edge_index[1][j] in self.x_new:
                        row.append(i)
                        col.append(self.data.edge_index[1][j])
        row = torch.tensor(row)
        row = row.to(self.device)
        col = torch.tensor(col)
        col = col.to(self.device)
        self.adj = SparseTensor(row=row, col=col, sparse_sizes=(len(self.x_new), len(self.x_new))).to(self.device) #это edge_index для train
        row2 = ((row.tolist()))
        col2 = ((col.tolist()))

        self.a = []
        self.a.append(row2)
        self.a.append(col2)
        self.a = torch.tensor(self.a,dtype=torch.long) #это edge_index для train
        
        self.walks_per_node = walks_per_node
        
        super(RWSampler, self).__init__()
    
    def pos_sample(self,batch):
        context_size = 10 if self.walk_length>=10 else self.walk_length
        batch = batch.repeat(self.walks_per_node) #так как в лоадере мы не меняли размер батча, по дефолту он раве 1, а значит мы повторили walks_per_node раз одну вершину 
        rowptr,col,_=self.adj.csr()
        rowptr = rowptr.to(self.device)
        col = col.to(self.device)       
        rw = random_walk(rowptr, col, batch,  self.walk_length, self.p, self.q) #построили по нашим row,col. по одному рандом волку размером walk_length из каждой вершины в батче 
        if not isinstance(rw, torch.Tensor):
            rw = rw[0]
        walks = []
        num_walks_per_rw = 1 + self.walk_length + 1 - context_size
        for j in range(num_walks_per_rw):
             walks.append(rw[:, j:j + context_size]) #теперь у нас внутри walks лежат 12 матриц размерам 10*10
        return  torch.cat(walks, dim=0)
    

    def neg_sample(self,batch):
        len_batch = len(batch)
        num_negative_samples = 1
        batch = batch.repeat(self.walks_per_node * num_negative_samples) 
        neg_batch=batched_negative_sampling(self.a, batch, num_neg_samples=num_negative_samples)
        neg_batch = neg_batch%len_batch
        return neg_batch
    
    
    def sample(self,batch):
        if not isinstance(batch, torch.Tensor):
            batch = torch.tensor(batch, dtype=torch.long).to(self.device)
        return self.pos_sample(batch),self.neg_sample(batch)
