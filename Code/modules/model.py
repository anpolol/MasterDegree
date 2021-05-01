import torch
from torch_geometric.nn import GCNConv, SAGEConv,GATConv, SGConv, ChebConv 
import torch.nn.functional as F
from torch_geometric.data import NeighborSampler
from datetime import datetime
class Net(torch.nn.Module):
    def __init__(self, dataset, device,loss_function,mode='unsupervised',conv='GCN',hidden_layer=64,out_layer =128,dropout = 0,num_layers=2):
        super(Net, self).__init__()
        self.mode = mode
        self.conv=conv
        self.num_layers = num_layers
        self.num_features = dataset.num_features
        self.num_classes = dataset.num_classes
        self.data = dataset[0]
        #print(dataset.num_features)
        self.loss_function = loss_function
        self.convs = torch.nn.ModuleList()
        self.hidden_layer =hidden_layer
        self.out_layer = out_layer
        self.dropout = dropout
        self.device=device
    
        if self.mode=='unsupervised':
            out_channels = self.out_layer
        elif self.mode=='supervised':
            out_channels = self.num_classes
        if self.conv == 'GCN':
            if self.num_layers == 1:
                self.convs.append(GCNConv(self.num_features, out_channels))
            else:
                self.convs.append(GCNConv(self.num_features, self.hidden_layer))
                for i in range(1,self.num_layers-1):
                    self.convs.append(GCNConv(self.hidden_layer, self.hidden_layer))
                self.convs.append(GCNConv(self.hidden_layer, out_channels))
        elif self.conv == 'SAGE':
            
            if self.num_layers == 1:
                self.convs.append(SAGEConv(self.num_features, out_channels))
            else:
                self.convs.append(SAGEConv(self.num_features, self.hidden_layer))
                for i in range(1,self.num_layers-1):
                    self.convs.append(SAGEConv(self.hidden_layer, self.hidden_layer))
                self.convs.append(SAGEConv(self.hidden_layer, out_channels))            
        elif self.conv == 'GAT':
            if self.num_layers == 1:
                self.convs.append(GATConv(self.num_features, out_channels))
            else: 
                self.convs.append(GATConv(self.num_features, self.hidden_layer))
                for i in range(1,self.num_layers-1):
                    self.convs.append(GATConv(self.hidden_layer, self.hidden_layer))
                self.convs.append(GATConv(self.hidden_layer, out_channels))
                
        if loss_function["loss var"] == "Random Walks":
            self.loss = self.lossRandomWalks
        elif loss_function["loss var"] == "Context Matrix":
            self.loss = self.lossContextMatrix
        elif loss_function["loss var"] == "Factorization":
            self.loss = self.lossFactorization
        elif loss_function["loss var"] == "Laplacian EigenMaps":
            self.loss = self.lossLaplacianEigenMaps
        self.reset_parameters()
            
    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
                   
    def forward(self,x,adjs):
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[:size[1]]  # Target nodes are always placed first.
            x = self.convs[i]((x,x_target), edge_index)
            if i != self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        if self.mode=='unsupervised':
            return x 
        elif self.mode=='supervised':
            return x.log_softmax(dim=1)
        
    def inference(self,data,dp=0): 
        
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i != self.num_layers - 1:
                x = x.relu()
                x = F.dropout(x, p=dp, training=self.training)
        if self.mode=='unsupervised':
            return x
        elif self.mode=='supervised':
            return x.log_softmax(dim=-1)       
              
    def lossRandomWalks(self,out, PosNegSamples):
        (pos_rw,neg_rw) = PosNegSamples
        pos_rw,neg_rw = pos_rw.type(torch.LongTensor).to(self.device),neg_rw.type(torch.LongTensor).to(self.device)
        # Positive loss.
        pos_loss=0
        start, rest = pos_rw[:, 0], pos_rw[:, 1:].contiguous()
        h_start = out[start].view(pos_rw.size(0), 1,self.out_layer)
        h_rest = out[rest.view(-1)].view(pos_rw.size(0), -1,self.out_layer)
        dot = (h_start * h_rest).sum(dim=-1).view(-1)
        pos_loss = -torch.log(torch.sigmoid(dot)).mean()
        
       # print('dot',dot.device)
            # Negative loss
        start, rest = neg_rw[:, 0], neg_rw[:, 1:].contiguous()
        h_start =out[start].view(neg_rw.size(0), 1,self.out_layer)
        h_rest =  out[rest.view(-1)].view(neg_rw.size(0), -1,self.out_layer)
        dot = (h_start * h_rest).sum(dim=-1).view(-1)
        neg_loss = -torch.log(torch.sigmoid((-1)*dot)).mean()

        return pos_loss + neg_loss
    
        
    def lossContextMatrix(self,out, PosNegSamples):
        (pos_rw,neg_rw) = PosNegSamples
            # Negative loss
        pos_rw=pos_rw.to(self.device)
        neg_rw = neg_rw.to(self.device)
        start, rest = neg_rw[:, 0].type(torch.LongTensor), neg_rw[:, 1:].type(torch.LongTensor).contiguous()
        h_start =out[start].view(neg_rw.size(0), 1,self.out_layer)
        
        h_rest =  out[rest.view(-1)].view(neg_rw.size(0), -1,self.out_layer)
        dot = (h_start * h_rest).sum(dim=-1).view(-1)
        neg_loss = -torch.log(torch.sigmoid((-1)*dot)).mean()
            # Positive loss.
        start, rest = pos_rw[:, 0].type(torch.LongTensor), pos_rw[:, 1].type(torch.LongTensor).contiguous()
        weight = pos_rw[:,2]
        h_start = out[start].view(pos_rw.size(0), 1,self.out_layer)

        h_rest = out[rest.view(-1)].view(pos_rw.size(0), -1,self.out_layer)
        dot = weight*((h_start * h_rest).sum(dim=-1)).view(-1)
        if self.loss_function["Name"] == "LINE":
            pos_loss = -2*(weight*torch.log(torch.sigmoid(dot))).mean()
        elif self.loss_function["Name"].split('_')[0] == "VERSE" or self.loss_function["Name"] == "APP":
            pos_loss = -(weight*torch.log(torch.sigmoid(dot))).mean() 

        return pos_loss + neg_loss
    def lossFactorization(self,out,S):
        S=S.to(self.device)
        lmbda=0.1
        loss = 0.5*sum(sum((S- torch.matmul(out,out.t())) *(S- torch.matmul(out,out.t())))) + 0.5*lmbda*sum(sum(out*out))
        return loss
    def lossLaplacianEigenMaps(self,out,A):
        dd=torch.device('cuda',1)
        L = (torch.diag(sum(A)) -A).type(torch.FloatTensor).to(dd)
        out_tr = out.t().to(dd) 
        loss = torch.trace(torch.matmul(torch.matmul(out_tr,L) ,out))
        return loss
        
    #loss function for supervised mode   

    def loss_sup(self, pred, label):
        return F.nll_loss(pred, label)