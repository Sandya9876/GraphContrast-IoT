import copy
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from src.augmentations import augment_batch
from src.losses import info_nce


def train_model(model, X_train, X_val, edge_index, edge_weight, cfg, device):
    tcfg = cfg['training']
    train_loader = DataLoader(TensorDataset(torch.tensor(X_train)), batch_size=tcfg['batch_size'], shuffle=True, drop_last=True)
    val_loader = DataLoader(TensorDataset(torch.tensor(X_val)), batch_size=tcfg['batch_size'], shuffle=False)
    opt = torch.optim.Adam(model.parameters(), lr=tcfg['learning_rate'])
    best, best_state, waits = float('inf'), copy.deepcopy(model.state_dict()), 0
    losses = []
    edge_index = edge_index.to(device); edge_weight = edge_weight.to(device)
    for epoch in range(1, tcfg['epochs']+1):
        model.train(); total = 0.0; n = 0
        for (xb,) in tqdm(train_loader, desc=f'Epoch {epoch}', leave=False):
            xb = xb.to(device)
            x1, ew1 = augment_batch(xb, edge_weight, cfg)
            x2, ew2 = augment_batch(xb, edge_weight, cfg)
            z1 = model(x1, edge_index, ew1); z2 = model(x2, edge_index, ew2)
            loss = info_nce(z1, z2, tcfg['temperature'])
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item()*xb.size(0); n += xb.size(0)
        avg = total/max(n,1); losses.append(avg)
        val_loss = _val_loss(model, val_loader, edge_index, edge_weight, cfg, device)
        if val_loss < best:
            best, best_state, waits = val_loss, copy.deepcopy(model.state_dict()), 0
        else:
            waits += 1
            if waits >= tcfg.get('patience',10): break
    model.load_state_dict(best_state)
    return model, losses

@torch.no_grad()
def _val_loss(model, loader, edge_index, edge_weight, cfg, device):
    model.eval(); total=0; n=0
    for (xb,) in loader:
        if xb.size(0) < 2: continue
        xb=xb.to(device)
        x1,ew1=augment_batch(xb,edge_weight,cfg); x2,ew2=augment_batch(xb,edge_weight,cfg)
        loss=info_nce(model(x1,edge_index,ew1), model(x2,edge_index,ew2), cfg['training']['temperature'])
        total += loss.item()*xb.size(0); n += xb.size(0)
    return total/max(n,1)
