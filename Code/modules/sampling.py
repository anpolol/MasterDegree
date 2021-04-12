import torch
from torch_cluster import random_walk
from torch_geometric.utils import structured_negative_sampling,batched_negative_sampling
import numpy as np
from modules.negativeSampling import NegativeSampler
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
from torch_sparse import SparseTensor
from torch.utils.data import DataLoader
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_cluster import random_walk
try:
    import torch_cluster  # noqa
    RW = torch.ops.torch_cluster.random_walk
except ImportError:
    RW = None
from functools import reduce
from torch_geometric.utils import subgraph 
class Sampler():
    def __init__(self, data,device, mask,loss_info,**kwargs):
        self.data = data
        self.device = 'cpu'
        self.mask = mask
        self.loss = loss_info
        self.num_negative_samples=1
        self.NS = NegativeSampler(self.data, self.device)
        if self.loss["loss var"] == "Random Walks":
            self.p=self.loss["p"]
            self.q=self.loss["q"]
            self.walk_length =self.loss["walk length"]
            self.walks_per_node = self.loss["walks per node"]
            self.context_size = self.loss["context size"] if self.walk_length>=self.loss["context size"] else self.walk_length
            self.num_negative_samples = self.loss["num negative samples"]
            self.pos_sample = self.pos_sample_rw
            self.neg_sample = self.neg_sample_rw
        elif self.loss["loss var"] == "Context Matrix":
            self.pos_sample = self.pos_sample_adj
            self.neg_sample = self.neg_sample_adj
        elif self.loss["loss var"] == "Factorization":
            pass
        super(Sampler, self).__init__()
    
    def edge_index_to_adj_train(self,mask,batch): 
        x_new=(torch.tensor(np.where(mask==True)[0],dtype=torch.int32))
        A = torch.zeros((len(x_new),len(x_new)),dtype=torch.long)
      #  print(len(x_new),x_new)
        for j,i in enumerate(self.data.edge_index[0]):
            if i in x_new:
                if self.data.edge_index[1][j] in x_new:
                    A[i%len(batch)][self.data.edge_index[1][j]%len(batch)]=1 
        return A      
    
    def pos_sample_rw(self,batch):
        len_batch = len(batch) 
        nodes = batch.numpy().tolist()
        a,_ = subgraph(nodes, self.data.edge_index)
        row,col=a 
        row = row.to(self.device)
        col = col.to(self.device) 
        start  = torch.tensor(list(set(row.tolist()) & set(col.tolist()) & set(batch.tolist())),dtype=torch.long)
        start = start.repeat(self.walks_per_node)
       
        if self.loss['Name'] == 'Node2Vec':
            adj = SparseTensor(row=row%len_batch, col=col%len_batch, sparse_sizes=(len_batch, len_batch))
            
            rowptr, col, _ = adj.csr()
            rw = RW(rowptr, col, start%len_batch, self.walk_length, self.p, self.q)
        else:
            rw = random_walk(row, col, start,  walk_length = self.walk_length) #построили по нашим row,col. по одному рандом волку размером walk_length из каждой вершины в батче 
        if not isinstance(rw, torch.Tensor):
            rw = rw[0]
        walks = []
        num_walks_per_rw = 1 + self.walk_length + 1 - self.context_size
        
        for j in range(num_walks_per_rw):
             walks.append(rw[:, j:j + self.context_size]) #теперь у нас внутри walks лежат 12 матриц размерам 10*1
        return  (torch.cat(walks, dim=0)%len_batch)
    


    def neg_sample_rw(self,batch):
        len_batch = len(batch)
        a,_=subgraph(batch.tolist(),self.data.edge_index)
        batch = batch.repeat(self.walks_per_node * self.num_negative_samples) 
        #print(c, batch,self.num_negative_samples)
        neg_batch = self.NS.negative_sampling(batch,num_negative_samples = self.num_negative_samples)
        return neg_batch%len_batch
    
    def pos_sample_adj(self,batch):
        batch = batch.tolist()
        pos_batch=[]
        mask = torch.tensor([False]*len(self.data.x))
        mask[batch] = True
        
        if self.loss["C"] == "Adj":
                A = self.edge_index_to_adj_train(mask,batch)
        elif self.loss["C"] == "PPR":
                Adg = self.edge_index_to_adj_train(mask,batch).type(torch.FloatTensor)
                invD =torch.diag(1/sum(Adg.t()))
                invD[torch.isinf(invD)] = 0
                alpha = 0.7
                A = ((1-alpha)*torch.inverse(torch.diag(torch.ones(len(Adg))) - alpha*torch.matmul(invD,Adg)))
             
        for x in batch:
            for j in range(len(A)):
                if A[x%len(batch)][j] != torch.tensor(0):
                    pos_batch.append([int(x%677),int(j),A[x%len(batch)][j]])
        return torch.tensor(pos_batch)

    def neg_sample_adj(self,batch):
        len_batch = len(batch)
        a,_=subgraph(batch.tolist(),self.data.edge_index)
        neg_batch=self.NS.negative_sampling(batch,num_negative_samples=self.num_negative_samples)
        return neg_batch%len_batch
   
    def sample(self,batch):
        if not isinstance(batch, torch.Tensor):
            batch = torch.tensor(batch, dtype=torch.long).to(self.device)
            
        if self.loss["loss var"] == "Factorization":
            if self.loss["C"] == "Adj":
                mask = torch.tensor([False]*len(self.data.x))
                mask[batch] = True
                C=  self.edge_index_to_adj_train(mask)
                
            elif self.loss["C"] == "Katz":
                mask = torch.tensor([False]*len(self.data.x))
                mask[batch] = True
                A =  self.edge_index_to_adj_train(mask)#.type(torch.FloatTensor)
                betta = 0.1
                I = torch.ones
                C = betta*torch.inverse((torch.diag(torch.ones(len(A))) - betta*A)) * A
                
            elif self.loss["C"] == "RPR":
                mask = torch.tensor([False]*len(self.data.x))
                mask[batch] = True
                Adg = self.edge_index_to_adj_train(mask).type(torch.FloatTensor)
                invD =torch.diag(1/sum(Adg.t()))
                invD[torch.isinf(invD)] = 0
                alpha = 0.7
                C = ((1-alpha)*torch.inverse(torch.diag(torch.ones(len(Adg))) - alpha*torch.matmul(invD,Adg)))
            return C
        else:
            return (self.pos_sample(batch),self.neg_sample(batch))

   