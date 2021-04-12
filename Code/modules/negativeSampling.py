import torch
from torch_cluster import random_walk
from torch_geometric.utils import structured_negative_sampling,batched_negative_sampling
import numpy as np
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_sparse import SparseTensor
from torch.utils.data import DataLoader
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_cluster import random_walk
from functools import reduce
from torch_geometric.utils import subgraph 
import random 
class NegativeSampler():
    def __init__(self, data,device,**kwargs):
        self.data = data
        self.device = 'cpu'
        super(NegativeSampler, self).__init__()
    
    def edge_index_to_train(self,mask):
        row=[]
        col =[]
        x_new=(torch.tensor(np.where(mask==True)[0],dtype=torch.int32))
        for j, i in enumerate(self.data.edge_index[0]):
            if i in x_new:
                if self.data.edge_index[1][j] in x_new:
                        row.append(i)
                        col.append(self.data.edge_index[1][j])
        row = torch.tensor(row)
        row = row.to(self.device)
        col = torch.tensor(col)
        col = col.to(self.device)
        
        adj = SparseTensor(row=row, col=col, sparse_sizes=(len(x_new), len(x_new))).to(self.device) #это edge_index для train
        
        row2 = row.tolist()
        col2 = col.tolist()
        a = []
        a.append(row2)
        a.append(col2)
        a = torch.tensor(a,dtype=torch.long) #это edge_index для train
        
        return adj, a   
    def not_less_than(self,k, l):
        if len(l) == 0:
            return l
        if len(l) >= k:
            return random.choices(l,k=k)#l[:k]
        return not_less_than(k, l*2)
    def adj_list(self,edge_index):
        f = dict()
        for x in list(zip(edge_index[0].tolist(), edge_index[1].tolist())):
            if (x[0] in f):
                f[x[0]].append(x[1]) 
            else:
                f[x[0]] = [x[1]]
        return f
    def torch_list(self,adj_list):
        line = list()
        other_line = list()
        for k, v in adj_list.items(): 
            line += [k] * len(v)
            other_line += v
        return torch.transpose((torch.tensor([line, other_line])),0,1)
    def negative_sampling(self, batch, num_negative_samples):
       # mask = torch.tensor([False]*len(self.data.x))
        #mask[batch] = True
        #_,a = self.edge_index_to_train(mask)
        a,_ = subgraph(batch,self.data.edge_index)
        f = self.adj_list(a)
        g = dict()
        l = batch.tolist()
        for e in l:
            g[e] = l
        for k, v in f.items():
            g[k] = list(set(l) - set(v))
        for k, v in g.items(): 
            g[k] = self.not_less_than(num_negative_samples, g[k])
        return self.torch_list(g)
 