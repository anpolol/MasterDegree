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
from functools import reduce
class Sampler():
    def __init__(self, data,device, mask,loss_info,**kwargs):
        self.data = data
        self.device = 'cpu'
        self.mask = mask
        self.loss = loss_info
        self.num_negative_samples=1
        
        if self.loss["loss var"] == "Random Walks":
            self.p=self.loss["p"]
            self.q=self.loss["q"]
            self.walk_length =self.loss["walk length"]
            self.walks_per_node = self.loss["walks per node"]
            self.context_size = self.loss["context size"] if self.walk_length>=self.loss["context size"] else self.walk_length
            self.num_negative_samples = self.loss["num negative samples"]
            self.pos_sample = self.pos_sample_rw
            self.neg_sample = self.neg_sample_rw
            self.adj, self.a = self.edge_index_to_train(self.mask)
        elif self.loss["loss var"] == "Context Matrix":
            self.pos_sample = self.pos_sample_adj
            self.neg_sample = self.neg_sample_adj
        elif self.loss["loss var"] == "Factorization":
            pass
        super(Sampler, self).__init__()
    
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
    def edge_index_to_adj_train(self,mask): 
        x_new=(torch.tensor(np.where(mask==True)[0],dtype=torch.int32))
        A = torch.zeros((len(x_new),len(x_new)),dtype=torch.long)
        for j,i in enumerate(self.data.edge_index[0]):
            if i in x_new:
                if self.data.edge_index[1][j] in x_new:
                    A[i][self.data.edge_index[1][j]]=1 
        return A      
    
    def pos_sample_rw(self,batch):
        batch = batch.repeat(self.walks_per_node) #так как в лоадере мы не меняли размер батча, по дефолту он раве 1, а значит мы повторили walks_per_node раз одну вершину 
        mask = torch.tensor([False]*len(self.data.x))
        mask[batch] = True
        adj,_ = self.edge_index_to_train(mask)
        rowptr,col,_=adj.csr()
        rowptr = rowptr.to(self.device)
        col = col.to(self.device)       
        rw = random_walk(rowptr, col, batch,  self.walk_length, self.p, self.q) #построили по нашим row,col. по одному рандом волку размером walk_length из каждой вершины в батче 
        if not isinstance(rw, torch.Tensor):
            rw = rw[0]
        walks = []
        num_walks_per_rw = 1 + self.walk_length + 1 - self.context_size
        
        for j in range(num_walks_per_rw):
             walks.append(rw[:, j:j + self.context_size]) #теперь у нас внутри walks лежат 12 матриц размерам 10*1
        return  torch.cat(walks, dim=0)

    def neg_sample_rw(self,batch):
        len_batch = len(batch)
        mask = torch.tensor([False]*len(self.data.x))
        mask[batch] = True
        _,c=self.edge_index_to_train(mask)
        batch = batch.repeat(self.walks_per_node * self.num_negative_samples) 
        neg_batch=batched_negative_sampling(c, batch, num_neg_samples=self.num_negative_samples)
        neg_batch = neg_batch%len_batch
        return neg_batch
    
    def pos_sample_adj(self,batch):
        batch = batch.tolist()
        pos_batch=[]
        mask = torch.tensor([False]*len(self.data.x))
        mask[batch] = True
        
        if self.loss["C"] == "Adj":
                A = self.edge_index_to_adj_train(mask)
        elif self.loss["C"] == "PPR":
                Adg = self.edge_index_to_adj_train(mask).type(torch.FloatTensor)
                invD =torch.diag(1/sum(Adg.t()))
                invD[torch.isinf(invD)] = 0
                alpha = 0.7
                A = ((1-alpha)*torch.inverse(torch.diag(torch.ones(len(Adg))) - alpha*torch.matmul(invD,Adg)))
             
        for x in batch:
            for j in range(len(A)):
                if A[x][j] != torch.tensor(0):
                    pos_batch.append([int(x),int(j),A[x][j]])
        return torch.tensor(pos_batch)

    def neg_sample_adj(self,batch):
        len_batch = len(batch)
        mask = torch.tensor([False]*len(self.data.x))
        mask[batch] = True
        _,c=self.edge_index_to_train(mask)
        neg_batch=batched_negative_sampling(c, batch,num_neg_samples=self.num_negative_samples)
        neg_batch = neg_batch%len_batch
        return neg_batch
     
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

   