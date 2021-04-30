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
from datetime import datetime
import random
import math
import collections
import abc
try:
    import torch_cluster  # noqa
    RW = torch.ops.torch_cluster.random_walk
except ImportError:
    RW = None
from functools import reduce
from torch_geometric.utils import subgraph 

class Sampler():
    def __init__(self, data,device, mask,loss_info,**kwargs):
        self.device = device
        self.data = data.to(self.device)
        self.mask = mask
        self.NS = NegativeSampler(self.data, self.device)
        self.loss = loss_info
        super(Sampler, self).__init__()
    def edge_index_to_adj_train(self,batch): 
        x_new=torch.sort(batch).values
       # долго работает наверное из-за .nonzero(as_tuple =True)         
        d_2=datetime.now()        
        x_map = dict()
        x_new = x_new.tolist()
        for j,i in enumerate(x_new):
            x_map[i] = j
        A = torch.zeros((len(x_new),len(x_new)),dtype=torch.long)#.to(self.device)
        edge_index_0 = self.data.edge_index[0].tolist()
        edge_index_1 = self.data.edge_index[1].tolist()
        for j,i in enumerate(edge_index_0):
            if i in x_new:
                if edge_index_1[j] in x_new:
                    A[x_map[i]][x_map[edge_index_1[j]]]=1
        
        return A,x_map
    def edge_index_to_adj_train_old(self,mask,batch): 
        x_new=(torch.tensor(np.where(mask==True)[0],dtype=torch.int32))
        A = torch.zeros((len(x_new),len(x_new)),dtype=torch.long)
        
        edge_index_0=self.data.edge_index[0].to('cpu')
        edge_index_1=self.data.edge_index[1].to('cpu')

        for j,i in enumerate(edge_index_0):
            if i in x_new:
                if edge_index_1[j] in x_new:
                    A[i][edge_index_1[j]]=1
                    #A[i%len(batch)][edge_index_1[j]%len(batch)]=1 
       # x_new=(batch)#(torch.tensor(np.where(mask.cpu()==True)[0],dtype=torch.int32))
       # A = torch.zeros((len(x_new),len(x_new)),dtype=torch.long).to(self.device)
       # for j,i in enumerate(x_new):
       #     for k in ((self.data.edge_index[0] == i).nonzero(as_tuple=True)[0]):
        #        if self.data.edge_index[1][k] in x_new:
         #           A[j][(x_new==self.data.edge_index[1][k]).nonzero(as_tuple =True)[0]] = 1    
        return A      
    @abc.abstractmethod
    def sample(self,batch,**kwargs):
         pass
     
    
class SamplerWithNegSamples(Sampler):
    def __init__(self,data,device,mask,loss_info,**kwargs):
        Sampler.__init__(self,data,device,mask,loss_info,**kwargs)
        self.num_negative_samples = self.loss["num_negative_samples"]
    def sample(self,batch):
        if not isinstance(batch, torch.Tensor):
            batch = torch.tensor(batch, dtype=torch.long).to(self.device)
        return (self.pos_sample(batch),self.neg_sample(batch))
    @abc.abstractmethod
    def pos_sample(self,batch):
        pass 
    def neg_sample(self,batch):
        len_batch = len(batch)
        a,_=subgraph(batch.tolist(),self.data.edge_index)
        neg_batch=self.NS.negative_sampling(batch,num_negative_samples=self.num_negative_samples)
        return neg_batch#%len_batch
        
class SamplerRandomWalk(SamplerWithNegSamples):
    def __init__(self, data,device, mask,loss_info,**kwargs):
            SamplerWithNegSamples.__init__(self, data,device, mask,loss_info,**kwargs)
            self.loss = loss_info
            self.p=self.loss["p"]
            self.q=self.loss["q"]
            self.walk_length =self.loss["walk_length"]
            self.walks_per_node = self.loss["walks_per_node"]
            self.context_size = self.loss["context_size"] if self.walk_length>=self.loss["context_size"] else self.walk_length
    def neg_sample(self,batch):
        len_batch = len(batch)
        a,_=subgraph(batch.tolist(),self.data.edge_index)
        batch = batch.repeat(self.walks_per_node * self.num_negative_samples) 
        #print(c, batch,self.num_negative_samples)
        neg_batch = self.NS.negative_sampling(batch,num_negative_samples = self.num_negative_samples)
        return neg_batch%len_batch
    def pos_sample(self,batch):
        d = datetime.now()
        len_batch = len(batch)
        a,_ = subgraph(batch, self.data.edge_index)
        row,col=a 
        row = row
        col = col 
        d1 = datetime.now()
        adj = SparseTensor(row=row, col=col, sparse_sizes=(len_batch, len_batch))
            
        rowptr, col, _ = adj.csr()
        d2 = datetime.now()
        start = batch.repeat(self.walks_per_node).to(self.device)
        rw = RW(rowptr, col, start, self.walk_length, self.p, self.q)

        if not isinstance(rw, torch.Tensor):
            rw = rw[0]
        walks = []
        num_walks_per_rw = 1 + self.walk_length + 1 - self.context_size
        
        for j in range(num_walks_per_rw):
             walks.append(rw[:, j:j + self.context_size]) #теперь у нас внутри walks лежат 12 матриц размерам 10*1
        return  (torch.cat(walks, dim=0)).to(self.device)
    
    
class SamplerContextMatrix(SamplerWithNegSamples):
    def __init__(self, data,device, mask,loss_info,**kwargs):
            SamplerWithNegSamples.__init__(self, data,device, mask,loss_info,**kwargs)
            self.loss = loss_info
            if self.loss["C"] == "PPR":
                self.alpha = self.loss["alpha"]
    def pos_sample(self,batch,**kwargs):
        d_pb =datetime.now()
        batch = batch
        pos_batch=[]
        d = datetime.now()
        if self.loss["C"] == "Adj" and self.loss["Name"] == "LINE":
                A,x_map = self.edge_index_to_adj_train(batch)
        elif self.loss["C"] == "Adj" and self.loss["Name"] == "VERSE_Adj":
                Adj,x_map = self.edge_index_to_adj_train(batch)
                A = (Adj / sum(Adj)).t()
                A[torch.isinf(A)] = 0
                A[torch.isnan(A)] = 0
        elif self.loss["C"] == "SR":
                Adj, _ = subgraph(batch,self.data.edge_index) 
                row,col= Adj 
                row = row.to(self.device)
                col = col.to(self.device)
                ASparse = SparseTensor(row=row, col=col, sparse_sizes=(len(batch), len(batch)))
                r = 200 
                length = list(map(lambda x: x*int(r/100), [22,17,14,10,8,6,5,4,3,11]))
                mask = []
                for i, l in enumerate(length):
                    mask1 = torch.zeros([l,10])
                    mask1.t()[:(i+1)] = 1
                    mask.append(mask1)
                mask = torch.cat(mask)
                mask_new = 1 - mask
                A = self.find_sim_rank_for_batch_torch(batch,ASparse,self.device,mask,mask_new,r)
                        
        elif self.loss["C"] == "PPR":
                    Adg,x_map = self.edge_index_to_adj_train(batch).type(torch.FloatTensor)
                    invD =torch.diag(1/sum(Adg.t()))
                    invD[torch.isinf(invD)] = 0
                    alpha = self.alpha
                    A = ((1-alpha)*torch.inverse(torch.diag(torch.ones(len(Adg))) - alpha*torch.matmul(invD,Adg)))
       #cpu:
        pos_batch = [] 
        for x in batch:
            for j in range(batch):
                if A[x_map[x]][x_map[j]] != torch.tensor(0):
                    pos_batch.append([int(x),int(j),A[x][j]])
        return torch.tensor(pos_batch)
        
       # A=A.to(self.device)
        #t = len(A)
        #pos_batch = torch.Tensor( torch.nonzero(A).size(0), 3 ).to(self.device)
        #p = 0 
        #for f,x in enumerate(batch):
        #    for j in range(t):
        #        if A[f][j] != torch.tensor(0):
        #            pos_batch[p][0] = (f)
        #            pos_batch[p][1] =(j)
        #            pos_batch[p][2] = (A[f][j])
        #            p+=1
        #return pos_batch
   
    def find_sim_rank_for_batch_torch(self,batch,Adj,device,mask,mask_new,r):
                t = 10
                c = torch.sqrt(torch.tensor(0.6))
                ## approx with SARW
                batch = batch.to(device)
                Adj = Adj.to(device)
                SimRank = torch.zeros(len(batch),len(batch)).to(device)
                for u in batch:
                    
                    d = datetime.now()
                    for nei in batch:
                        prob_i =torch.zeros(t).to(device)
                        
                        d_rw = datetime.now()
                        pi_u = Adj.random_walk(u.repeat(r).flatten(), walk_length =t)
                        pi_v = Adj.random_walk(nei.repeat(r).flatten(), walk_length =t)
                        pi_u = pi_u[:,1:]
                        pi_v = pi_v[:,1:]
                       
                        pi_u = pi_u * mask - mask_new
                        pi_v = pi_v * mask - mask_new
                        

                        d_sr = datetime.now()            
                        a1 = pi_u == pi_v
                        a2 = pi_u != -1
                        a3 = pi_v != -1
                        a_to_compare = a1*a2*a3
                        SR = len(torch.unique((a_to_compare).nonzero().t()[0]))
                        SimRank[u][nei] = SR/r
                     
                return SimRank 

        
class SamplerFactorization(Sampler):
    def sample(self,batch,**kwargs):
            if not isinstance(batch, torch.Tensor):
                batch = torch.tensor(batch, dtype=torch.long).to(self.device)

            A,_ =  self.edge_index_to_adj_train(batch)
            if self.loss["loss var"] == "Factorization":
                if self.loss["C"] == "Adj":
                    C = A
                elif self.loss["C"] == "CN" :
                    C = torch.matmul(A,A)
                elif self.loss["C"] == "AA":
                    D = torch.diag(1/(sum(A) + sum(A.t()))) 
                    A = A.type(torch.FloatTensor)
                    D[torch.isinf(D)] = 0
                    D[torch.isnan(D)] = 0
                    C = torch.matmul(torch.matmul(A, D) ,A) 
                elif self.loss["C"] == "Katz":
                    betta =self.loss["betta"]
                    I = torch.ones
                    C = betta*torch.inverse((torch.diag(torch.ones(len(A))) - betta*A)) * A
                elif self.loss["C"] == "RPR":
                    alpha = self.loss["alpha"]
                    A = A.type(torch.FloatTensor)
                    invD =torch.diag(1/sum(A.t()))
                    invD[torch.isinf(invD)] = 0
                    C = ((1-alpha)*torch.inverse(torch.diag(torch.ones(len(A))) - alpha*torch.matmul(invD,A)))
                return C
            else:
                return A
            
class SamplerAPP(SamplerWithNegSamples):
    def __init__(self, data,device, mask,loss_info,**kwargs):
            SamplerWithNegSamples.__init__(self, data,device, mask,loss_info,**kwargs)
            self.device = torch.device('cuda',1)# if torch.cuda.is_available() else 'cpu')
            self.alpha = self.loss["alpha"]
            self.r = 200
            self.num_negative_samples *=10
    def sample(self,batch,**kwargs):
        if not isinstance(batch, torch.Tensor):
            batch = torch.tensor(batch, dtype=torch.long).to(self.device)
        return (self.pos_sample(batch),self.neg_sample(batch))
    def pos_sample(self,batch,**kwargs):
        d_pb =datetime.now()
        batch = batch.to(self.device)
        len_batch = len(batch)
        mask = torch.tensor([False]*len(self.data.x))
        mask[batch.tolist()] = True
        d = datetime.now()
       
        a,_ = subgraph(batch, self.data.edge_index)
        row,col=a 
        row = row.to(self.device)
        col = col.to(self.device) 
                    #start  = torch.tensor(list(set(row.tolist()) & set(col.tolist()) & set(batch.tolist())),dtype=torch.long)
        ASparse = SparseTensor(row=row, col=col, sparse_sizes=(len_batch, len_batch))
        
        pos_dict = self.find_PPR_approx(batch,ASparse,self.device,self.r,self.alpha)
        pos_batch=[]
        for pos_pair in pos_dict:
            pos_row  = list(pos_pair) 
            pos_row.append(pos_dict[pos_pair])
            pos_batch.append(pos_row)
        return torch.tensor(pos_batch)
   
    def find_PPR_approx(self,batch,Adj,device,r,alpha):
        N=math.ceil(math.log(1/(r*alpha),(1-alpha)))
        length = list(map(lambda x: int((1-alpha)**x*alpha *(r)), list(range(N-1))))
        r = sum(length)
        dict_data = dict()
        device='cpu'
        for u in batch:
            d = datetime.now()
            pi_u = Adj.random_walk(u.to(self.device).repeat(r).flatten(), walk_length =N)
            split = torch.split(pi_u, length)
            pos_batch = []
            
            for i,seg in enumerate(split):
                pos_samples = collections.Counter(seg[:,(i+1)].tolist())
                for pos_sample in pos_samples:
                    if (int(seg[0][0]),int(pos_sample)) in dict_data:
                        dict_data[(int(seg[0][0]),int(pos_sample))] += pos_samples[pos_sample]
                    elif (int(pos_sample),int(seg[0][0])) in dict_data:
                        dict_data[(int(pos_sample),int(seg[0][0]))] += pos_samples[pos_sample]
                    else:
                        dict_data[(int(seg[0][0]),int(pos_sample))] = 1
        return dict_data 