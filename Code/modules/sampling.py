#http://localhost:8888/edit/Desktop/MasterDegree/Code/modules/sampling.py#Сэмплирование позитивных и негативных примеров  
import torch
from torch_cluster import random_walk
from torch_geometric.utils import negative_sampling
from torch_geometric.utils import structured_negative_sampling
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
    
class RWSampler(torch.utils.data.DataLoader):
    def __init__(self, data, mask,device,walk_length=20,p=1,q=1,**kwargs):
        self.data = data
        self.device = device
        self.mask = mask
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
        self.a = torch.tensor(self.a)
        node_idx = torch.arange(self.adj.sparse_size(0))
        node_idx.view(-1).tolist()
        super(RWSampler, self).__init__(node_idx, collate_fn=self.sample, **kwargs)

        
    
    def pos_sample(self,batch):
            
            walks_per_node = 10
            context_size = 10
            batch = batch.repeat(walks_per_node) #так как в лоадере мы не меняли размер батча, по дефолту он раве 1, а значит мы повторили 10 раз одну вершину 
            rowptr,col,_=self.adj.csr()
            rowptr = rowptr.to(self.device)
            col = col.to(self.device)

            rw = random_walk(rowptr, col, batch,  self.walk_length, self.p, self.q) #построили по нашим row,col. по одному рандом волку размером walk_length из каждой вершины в батче (в нашем случае это одна и та же вершина повторенная 10 раз)
            if not isinstance(rw, torch.Tensor):
                rw = rw[0]
            walks = []
            num_walks_per_rw = 1 + self.walk_length + 1 - context_size
            for j in range(num_walks_per_rw):
                walks.append(rw[:, j:j + context_size]) #теперь у нас внутри walks лежат 12 матриц размерам 10*10 
            return torch.cat(walks, dim=0) #Благодаря конкатенации теперь walks выглядит как матрица размером 120*10
    #в train_test мы проходим по загрузчику поэтапно, значит батч меняется от 1 до 140ой вершины, на каждую вершину мы получаем по матрице размером 120 на 10. 


    def neg_sample(self,batch):
        
        walks_per_node = 10
        num_negative_samples=1
        batch = batch.repeat(walks_per_node * num_negative_samples) 
        neg_sam = structured_negative_sampling(self.a)
        b = []
        b.append(neg_sam[0].tolist())
        c =neg_sam[2].tolist()
        b.append(c)
        b = ((np.array(b).transpose()))
        b = torch.tensor(b).type(torch.LongTensor)
        return b # матрица размером 43*2
    
    
    def sample(self,batch):
        if not isinstance(batch, torch.Tensor):
            batch = torch.tensor(batch, dtype=torch.long).to(self.device)
        return self.pos_sample(batch),self.neg_sample(batch)
  