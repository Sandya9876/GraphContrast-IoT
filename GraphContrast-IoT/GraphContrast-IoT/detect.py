import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

@torch.no_grad()
def embed(model, X, edge_index, edge_weight, batch_size, device):
    model.eval(); zs=[]
    loader=DataLoader(TensorDataset(torch.tensor(X)), batch_size=batch_size, shuffle=False)
    edge_index=edge_index.to(device); edge_weight=edge_weight.to(device)
    for (xb,) in loader:
        zs.append(model(xb.to(device), edge_index, edge_weight).cpu().numpy())
    return np.vstack(zs)

def fit_centroid_threshold(train_z, val_z, val_y=None, k=3.0):
    centroid = train_z.mean(axis=0)
    val_scores = np.linalg.norm(val_z-centroid, axis=1)
    normal_scores = val_scores[val_y==0] if val_y is not None and (val_y==0).any() else val_scores
    threshold = normal_scores.mean() + k*normal_scores.std()
    return centroid, float(threshold), val_scores

def score_embeddings(z, centroid, threshold):
    scores = np.linalg.norm(z-centroid, axis=1)
    pred = (scores > threshold).astype(int)
    return scores, pred
