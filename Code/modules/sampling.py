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
try:
    import torch_cluster  # noqa
    RW = torch.ops.torch_cluster.random_walk
except ImportError:
    RW = None
from functools import reduce
from torch_geometric.utils import subgraph 

class Sampler():
    def __init__(self, data,device, mask,loss_info,**kwargs):
        self.device = 'cpu'
        self.data = data.to(self.device)
        self.mask = mask
        self.NS = NegativeSampler(self.data, self.device)
        self.loss = loss_info
        super(Sampler, self).__init__()
    
    def edge_index_to_adj_train(self,mask,batch): 
        x_new=(torch.tensor(np.where(mask==True)[0],dtype=torch.int32))
        A = torch.zeros((len(x_new),len(x_new)),dtype=torch.long)
        
        edge_index_0=self.data.edge_index[0].to('cpu')
        edge_index_1=self.data.edge_index[1].to('cpu')

        for j,i in enumerate(edge_index_0):
            if i in x_new:
                if edge_index_1[j] in x_new:
                    A[i%len(batch)][edge_index_1[j]%len(batch)]=1 
       # x_new=(batch)#(torch.tensor(np.where(mask.cpu()==True)[0],dtype=torch.int32))
       # A = torch.zeros((len(x_new),len(x_new)),dtype=torch.long).to(self.device)
       # for j,i in enumerate(x_new):
       #     for k in ((self.data.edge_index[0] == i).nonzero(as_tuple=True)[0]):
        #        if self.data.edge_index[1][k] in x_new:
         #           A[j][(x_new==self.data.edge_index[1][k]).nonzero(as_tuple =True)[0]] = 1    
        return A      
    
    def sample(self,batch,**kwargs):
         pass
        
class SamplerRandomWalk(Sampler):
    def __init__(self, data,device, mask,loss_info,**kwargs):
            Sampler.__init__(self, data,device, mask,loss_info,**kwargs)
            self.loss = loss_info
            self.p=self.loss["p"]
            self.q=self.loss["q"]
            self.walk_length =self.loss["walk_length"]
            self.walks_per_node = self.loss["walks_per_node"]
            self.context_size = self.loss["context_size"] if self.walk_length>=self.loss["context_size"] else self.walk_length
            self.num_negative_samples = self.loss["num_negative_samples"]
           # super(Sampler,self,__init__)
    def sample(self,batch,**kwargs):
        if not isinstance(batch, torch.Tensor):
            batch = torch.tensor(batch, dtype=torch.long).to(self.device)
        return (self.pos_sample(batch),self.neg_sample(batch))
    def pos_sample(self,batch):
        d = datetime.now()
        device = torch.device('cuda')
        len_batch = len(batch) 
        nodes = batch.numpy().tolist()
        a,_ = subgraph(nodes, self.data.edge_index)
        row,col=a 
        row = row.to(device)
        col = col.to(device) 
        start  = torch.tensor(list(set(row.tolist()) & set(col.tolist()) & set(batch.tolist())),dtype=torch.long)
        start = start.repeat(self.walks_per_node).to(device)
        if self.loss['Name'] == 'Node2Vec':
            adj = SparseTensor(row=row%len_batch, col=col%len_batch, sparse_sizes=(len_batch, len_batch))
            
            rowptr, col, _ = adj.csr()
            #rw = adj.randint(start%len_batch) 
            rw = RW(rowptr, col, start%len_batch, self.walk_length, self.p, self.q)
        else:
            adj = SparseTensor(row=row%len_batch, col=col%len_batch, sparse_sizes=(len_batch, len_batch)).to('cuda')
            rw = adj.random_walk(start.flatten(), walk_length = self.walk_length)
            
            #rw = random_walk(row, col, start,  walk_length = self.walk_length)
        if not isinstance(rw, torch.Tensor):
            rw = rw[0]
        walks = []
        num_walks_per_rw = 1 + self.walk_length + 1 - self.context_size
        
        for j in range(num_walks_per_rw):
             walks.append(rw[:, j:j + self.context_size]) #теперь у нас внутри walks лежат 12 матриц размерам 10*1
        return  (torch.cat(walks, dim=0)%len_batch)
    
    def neg_sample(self,batch):
        len_batch = len(batch)
        a,_=subgraph(batch.tolist(),self.data.edge_index)
        batch = batch.repeat(self.walks_per_node * self.num_negative_samples) 
        #print(c, batch,self.num_negative_samples)
        neg_batch = self.NS.negative_sampling(batch,num_negative_samples = self.num_negative_samples)
        return neg_batch%len_batch
    
class SamplerContextMatrix(Sampler):
    def __init__(self, data,device, mask,loss_info,**kwargs):
            Sampler.__init__(self, data,device, mask,loss_info,**kwargs)
            self.loss = loss_info
            if self.loss["C"] == "PPR":
                self.alpha = kwargs["alpha"]
            self.num_negative_samples = self.loss["num_negative_samples"]
            self.num_negative_samples = self.loss["num_negative_samples"]
            #super(SamplerContextMatrix,self,__init__)
    def sample(self,batch,**kwargs):
        if not isinstance(batch, torch.Tensor):
            batch = torch.tensor(batch, dtype=torch.long).to(self.device)
        return (self.pos_sample(batch),self.neg_sample(batch))
    def pos_sample(self,batch,**kwargs):
        d_pb =datetime.now()
        device = torch.device('cuda')
        batch = batch
        pos_batch=[]
        mask = torch.tensor([False]*len(self.data.x))
        mask[batch.tolist()] = True
        d = datetime.now()
        if self.loss["C"] == "Adj" and self.loss["Name"] == "LINE":
                A = self.edge_index_to_adj_train(mask,batch.tolist())
        elif self.loss["C"] == "Adj" and self.loss["Name"] == "VERSE_Adj":
                Adj = self.edge_index_to_adj_train(mask,batch.tolist())
                A = (Adj / sum(Adj)).t()
                A[torch.isinf(A)] = 0
                A[torch.isnan(A)] = 0
        elif self.loss["C"] == "SR":
                #Adj = self.edge_index_to_adj_train(mask,batch)
                Adj, _ = subgraph(batch.tolist(),self.data.edge_index) 
                row,col= Adj 
                row = row.to(device)
                col = col.to(device)
                ASparse = SparseTensor(row=row%len(batch), col=col%len(batch), sparse_sizes=(len(batch), len(batch)))
            
                A = self.find_sim_rank_for_batch_torch(batch,ASparse,device)
                        
        elif self.loss["C"] == "PPR":
                Adg = self.edge_index_to_adj_train(mask,batch).type(torch.FloatTensor)
                invD =torch.diag(1/sum(Adg.t()))
                invD[torch.isinf(invD)] = 0
                alpha = self.alpha
                A = ((1-alpha)*torch.inverse(torch.diag(torch.ones(len(Adg))) - alpha*torch.matmul(invD,Adg)))
        
        elif self.loss["Name"] == "APP":
                pass #Implement
        #print('counting of adj matrix', datetime.now()-d)
        dd = datetime.now()
        for x in batch:
            for j in range(len(A)):
                if A[x%len(batch)][j] != torch.tensor(0):
                    pos_batch.append([int(x%len(batch)),int(j),A[x%len(batch)][j]]) 
        
       # print('pos batch sampling ', datetime.now() - d_pb)
       # print('converting adj matrix', datetime.now() - dd)
      #  p = 0 
        #for f,x in enumerate(batch):
         #   for j in range(t):
          #      if A[f][j] != torch.tensor(0):
           #         pos_batch[p][0] = (f)
            #        pos_batch[p][1] =(j)
             #       pos_batch[p][2] = (A[f][j])
              #      p+=1
        pos_batch  = torch.tensor(pos_batch)
        return pos_batch

    def neg_sample(self,batch):
        d_nb = datetime.now()
        len_batch = len(batch)
        a,_=subgraph(batch.tolist(),self.data.edge_index)
        neg_batch=self.NS.negative_sampling(batch,num_negative_samples=self.num_negative_samples)
       # print('neg batch sampling ', datetime.now() - d_nb)
        return neg_batch%len_batch
    def find_sim_rank_for_batch(self,batch,Adj):
                r = 100
                t = 10
                c = torch.sqrt(torch.tensor(0.6))
                ## approx with SARW
                
                A = torch.zeros(len(Adj),len(Adj)).to('cuda')
                for u in range(len(Adj)):
                    print(u)
                    d = datetime.now()
                    for nei in range(len(Adj)):
                        prob_i =np.zeros((t))
                        for k in range(r):
                            
                            pi_u = [u]
                            pi_v = [nei]
                            if len((Adj[pi_v[0]] == 1).nonzero(as_tuple=True)[0])>0 and len((Adj[pi_u[0]] == 1).nonzero(as_tuple=True)[0])>0:
                                i =0
                                pi_u.append(np.random.choice(((Adj[pi_u[i]] == 1).nonzero(as_tuple=True)[0])))
                                pi_v.append(np.random.choice(((Adj[pi_v[i]] == 1).nonzero(as_tuple=True)[0])))
                                if pi_u[i+1] == pi_v[i+1]:
                                    prob_i[i]+=1
                                    break
                                for i in range(1,t):
                                        pr = np.random.random()
                                        if pr > c:
                                            if len((Adj[pi_v[i]] == 1).nonzero(as_tuple=True)[0])>0 and len((Adj[pi_u[i]] == 1).nonzero(as_tuple=True)[0])>0:
                                                
                                                pi_u.append(np.random.choice(((Adj[pi_u[i]] == 1).nonzero(as_tuple=True)[0])))
                                                pi_v.append(np.random.choice(((Adj[pi_v[i]] == 1).nonzero(as_tuple=True)[0])))

                                                if pi_u[i+1] == pi_v[i+1]:
                                                    prob_i[i]+=1
                                                    break
                                            else:
                                                break
                                        else:
                                            break
                        prob_i = prob_i/r
                        A[u][nei] = sum(prob_i)
                return A
    def find_sim_rank_for_batch_torch(self,batch,Adj,device):
                r = 100
                t = 10
                c = torch.sqrt(torch.tensor(0.6))
                ## approx with SARW
                batch = batch.to(device)
                Adj = Adj.to(device)
                SimRank = torch.zeros(len(batch),len(batch)).to(device)
                print(len(batch))
                for u in batch:
                    d = datetime.now()
                    for nei in batch:
                        prob_i =torch.zeros(t).to('cuda')
                        
                        d_rw = datetime.now()
                        pi_u = Adj.random_walk(u.repeat(r).flatten(), walk_length =t)
                        pi_v = Adj.random_walk(nei.repeat(r).flatten(), walk_length =t)
                        pi_u = pi_u[:,1:]
                        pi_v = pi_v[:,1:]
                        #print('random walk counting',datetime.now()-d_rw)
                        #print(list(map(lambda x: self.f(x, c), pi_u)))
                        d_r = datetime.now()
                     #   b = torch.rand(pi_u.size())
                     #   b2 =  torch.rand(pi_v.size())
                        #print('random matrices counting',datetime.now()-d_r)
                        d_ind = datetime.now()
                     #   ind1=(b>c).nonzero()
                     #   ind2=(b2>c).nonzero()
                       # print('indices counting',datetime.now()-d_ind)
                        
                     #   d_pi = datetime.now()
                     #   for row_elem,col_elem in ind1:
                     #       pi_u[row_elem][col_elem:] =-1
    
                        
                     #   for row_elem,col_elem in ind2:
                     #       pi_v[row_elem][col_elem:]=-1 
                     
                        for j,row in enumerate(pi_u):
                            prob = torch.rand(t)
                            if torch.any(prob>c):
                                ind = (prob>c).nonzero()[0] 
                                row[ind:] = -1
                            
                        for j,row in enumerate(pi_v):
                            prob = torch.rand(t)
                            if torch.any(prob>c):
                                ind = (prob>c).nonzero()[0] 
                                row[ind:] = -1
                       # print('pi matrices improving',datetime.now()-d_ind)

                        d_sr = datetime.now()            
                        a1 = pi_u == pi_v
                        a2 = pi_u != -1
                        a3 = pi_v != -1
                        a_to_compare = a1*a2*a3
                        SR = len(torch.unique((a_to_compare).nonzero().t()[0]))
                        SimRank[u][nei] = SR/r
                        #print('sim ramk couning',datetime.now()-d_sr)
                    print(SimRank[u],datetime.now()-d)
                        

                return SimRank 

        
class SamplerFactorization(Sampler):

    def sample(self,batch,**kwargs):
            if not isinstance(batch, torch.Tensor):
                batch = torch.tensor(batch, dtype=torch.long).to(self.device)
            mask = torch.tensor([False]*len(self.data.x))
            mask[batch] = True
            A =  self.edge_index_to_adj_train(mask,batch)
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
                    A = self.edge_index_to_adj_train(mask,batch).type(torch.FloatTensor)
                    invD =torch.diag(1/sum(A.t()))
                    invD[torch.isinf(invD)] = 0
                    C = ((1-alpha)*torch.inverse(torch.diag(torch.ones(len(A))) - alpha*torch.matmul(invD,A)))
                return C
            else:
                return A
 

