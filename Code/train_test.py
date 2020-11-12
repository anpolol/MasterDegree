import torch
from torch_geometric.data import NeighborSampler
device = 'cpu'
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid
dataset =Planetoid(root='/tmp/Cora', name='Cora',transform=T.NormalizeFeatures())
data = dataset[0]
from sklearn.neural_network import MLPClassifier
x = data.x.to(device)
y = data.y.squeeze().to(device)

def train(model,data,optimizer,loader):
    model.train()        
    train_loader = NeighborSampler(data.edge_index, batch_size = 1078, node_idx=data.train_mask, sizes=[25, 10],
                               shuffle=True)
    total_loss = 0
    for _, n_id, adjs in train_loader:
        # `adjs` holds a list of `(edge_index, e_id, size)` tuples.
        adjs = [adj.to(device) for adj in adjs]
        out = model.forward(data.x[n_id].to(device), adjs)
        optimizer.zero_grad()
        for i, (pos_rw, neg_rw) in enumerate(loader):
            loss = model.loss(out, pos_rw.to(device), neg_rw.to(device))
            total_loss+=loss
    total_loss.backward(retain_graph=True)
    optimizer.step()      
    return total_loss /len(loader)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
@torch.no_grad()
def test(model,data):
    model.eval()
    out = model.inference(data.x,data)
    y_true = y.cpu().detach().numpy()
    clf = MLPClassifier(random_state=1, max_iter=2000).fit(out.cpu().detach().numpy(),y_true)
    y_pred = clf.predict(out.cpu().detach().numpy())
    #cm = confusion_matrix(y_true,y_pred)
    results = []
    for mask in [data.train_mask, data.val_mask, data.test_mask]:    
        results += [(clf.score(out.cpu().detach()[mask].numpy(), y.cpu().detach()[mask].numpy()))]
        
    return results
