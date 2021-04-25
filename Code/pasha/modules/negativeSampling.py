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
  
    def not_less_than(self,num_negative_samples, all_negative_samples):
        if len(all_negative_samples) == 0:
            return all_negative_samples
        if len(all_negative_samples) >= num_negative_samples:
            return random.choices(all_negative_samples,k=num_negative_samples)#l[:k]
        return not_less_than(num_negative_samples, all_negative_samples*2)
    def adj_list(self,edge_index): #считаем список рёбер из edge_index 
        Adj_list = dict()
        for x in list(zip(edge_index[0].tolist(), edge_index[1].tolist())):
            if (x[0] in Adj_list):
                Adj_list[x[0]].append(x[1]) 
            else:
                Adj_list[x[0]] = [x[1]]
        return Adj_list
    def torch_list(self,adj_list):
        line = list()
        other_line = list()
        for node, neghbors in adj_list.items(): 
            line += [node] * len(neghbors)
            other_line += neghbors
        return torch.transpose((torch.tensor([line, other_line])),0,1)
    def negative_sampling(self, batch, num_negative_samples):
       # mask = torch.tensor([False]*len(self.data.x))
        #mask[batch] = True
        #_,a = self.edge_index_to_train(mask)
        a,_ = subgraph(batch,self.data.edge_index)
        Adj = self.adj_list(a) 
        g = dict()
        batch = batch.tolist()
        for node in batch:
            g[node] = batch
        for node, neghbors in Adj.items():
            g[node] = list(set(batch) - set(neghbors)) #тут все элементы которые не являются соседянями, но при этом входят в батч 
        for node, neg_elem in g.items(): 
            g[node] = self.not_less_than(num_negative_samples, g[node]) #если просят конкретное число негативных примеров, надо либо обрезать либо дублировать 
        return self.torch_list(g)
 