import torch
from torch_geometric.nn import GCNConv, SAGEConv,GATConv, SGConv, ChebConv 
import torch.nn.functional as F
from torch_geometric.data import NeighborSampler
class Net(torch.nn.Module):
    def __init__(self, dataset, device,mode='unsupervised',conv='GCN',loss_function='PosNegSamples',hidden_layer=64,out_layer =128,dropout = 0,num_layers=2):
        super(Net, self).__init__()
        self.mode = mode
        self.conv=conv
        self.num_layers = num_layers
        self.num_features = dataset.num_features
        self.num_classes = dataset.num_classes
        self.data = dataset[0]
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
        elif self.conv == 'SGC':
            self.convs.append(SGConv(self.num_features, self.hidden_layer))
            for i in range(1,self.num_layers-1):
                self.convs.append(SGConv(self.hidden_layer, self.hidden_layer))
            self.convs.append(SGConv(self.hidden_layer, out_channels))
        elif self.conv == 'Cheb':
            self.convs.append(ChebConv(self.num_features, self.hidden_layer,K=2))
            for i in range(1,self.num_layers-1):
                self.convs.append(ChebConv(self.hidden_layer, self.hidden_layer))
            self.convs.append(ChebConv(self.hidden_layer, out_channels,K=2))
        if loss_function == "Random Walks":
            self.loss = self.lossRandomWalks
        elif loss_function == "Context Matrix":
            self.loss = self.lossContextMatrix
        elif loss_function == "Factorization":
            self.loss = self.lossFactorization
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
        pos_rw,neg_rw = pos_rw.to(self.device),neg_rw.to(self.device)
        # Positive loss.
        pos_loss=0
        start, rest = pos_rw[:, 0], pos_rw[:, 1:].contiguous()
        h_start = out[start].view(pos_rw.size(0), 1,self.out_layer)
        h_rest = out[rest.view(-1)].view(pos_rw.size(0), -1,self.out_layer)
        dot = (h_start * h_rest).sum(dim=-1).view(-1)
        pos_loss = -torch.log(torch.sigmoid(dot)).mean()
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
        start, rest = neg_rw[:, 0], neg_rw[:, 1:].contiguous()
        h_start =out[start].view(neg_rw.size(0), 1,self.out_layer)
        h_rest =  out[rest.view(-1)].view(neg_rw.size(0), -1,self.out_layer)
        dot = (h_start * h_rest).sum(dim=-1).view(-1)
        neg_loss = -torch.log(torch.sigmoid((-1)*dot)).mean()
            # Positive loss.
        pos_loss=0
        start, rest = pos_rw[:, 0].type(torch.LongTensor), pos_rw[:, 1].contiguous().type(torch.LongTensor)
        weight = pos_rw[:,2]
        #print(
        h_start = out[start].view(pos_rw.size(0), 1,self.out_layer)
        h_rest = out[rest.view(-1)].view(pos_rw.size(0), -1,self.out_layer)
        dot = weight*((h_start * h_rest).sum(dim=-1)).view(-1)
        pos_loss = -torch.log(torch.sigmoid(dot)).mean()
          
        return pos_loss + neg_loss
    def lossFactorization(self,out,A):
        lmbda=10
        loss = 0.5*sum(sum((A- torch.matmul(out,out.t())) *(A- torch.matmul(out,out.t())))) + 0.5*lmbda*sum(sum(out*out))
        return loss
        
    #loss function for supervised mode   
    def loss_sup(self, pred, label):
        return F.nll_loss(pred, label)
    