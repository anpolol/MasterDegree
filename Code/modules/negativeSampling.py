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
    def __init__(self, data,device='cpu',**kwargs):
        self.data = data
        self.device = 'cpu'
        super(NegativeSampler, self).__init__()
    def not_less_than(self,k, l):
        if len(l) == 0:
            return l
        if len(l) >= k:
            return random.choices(l,k=k)#l[:k]
        else:
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
 