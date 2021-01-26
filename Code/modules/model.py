import torch
from torch_geometric.nn import GCNConv, SAGEConv,GATConv
import torch.nn.functional as F
from torch_geometric.data import NeighborSampler
class Net(torch.nn.Module):
    def __init__(self, dataset, device,mode='unsupervised',conv='GCN',loss_function='DeepWalk',hidden_layer=256,out_layer =128):
        super(Net, self).__init__()
        self.mode = mode
        self.conv=conv
        self.num_layers = 2
        self.num_features = dataset.num_features
        self.num_classes = dataset.num_classes
        self.loss_function = loss_function
        self.convs = torch.nn.ModuleList()
        self.hidden_layer =hidden_layer
        self.out_layer = out_layer
        self.device=device
    
        if self.mode=='unsupervised':
            out_channels = self.out_layer
        elif self.mode=='supervised':
            out_channels = self.num_classes
        if self.conv == 'GCN':
            self.convs.append(GCNConv(self.num_features, self.hidden_layer))
            self.convs.append(GCNConv(self.hidden_layer, out_channels))
        if self.conv == 'SAGE':
            self.convs.append(SAGEConv(self.num_features, self.hidden_layer))
            self.convs.append(SAGEConv(self.hidden_layer, out_channels))
        if self.conv == 'GAT':
            self.convs.append(GATConv(self.num_features, self.hidden_layer))
            self.convs.append(GATConv(self.hidden_layer, out_channels))
                
    def forward(self,x,adjs):
        for i, (edge_index, _, size) in enumerate(adjs):
            x_target = x[:size[1]]  # Target nodes are always placed first.
            x = self.convs[i]((x,x_target), edge_index)
            if i != self.num_layers - 1:
                x = F.relu(x)
                x = F.dropout(x, p=0.5, training=self.training)
        if self.mode=='unsupervised':
            return x
        elif self.mode=='supervised':
            return x.log_softmax(dim=1)
        
    def inference(self,data): 
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_attr
        x = F.relu(self.convs[0](x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.convs[1](x, edge_index, edge_weight)
        if self.mode=='unsupervised':
            return x
        elif self.mode=='supervised':
            return x.log_softmax(dim=-1)       
              
    def loss(self,out, pos_rw,neg_rw):
        if self.loss_function == 'DeepWalk':
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
    #loss function for supervised mode   
    def loss_sup(self, pred, label):
        return F.nll_loss(pred, label)