from pathlib import Path
import numpy as np
import torch


def build_adjacency(train_windows, cfg):
    # train_windows: [B, N, W, 1]
    gcfg = cfg['graph']; X = train_windows[...,0]  # [B,N,W]
    N = X.shape[1]
    sensor_series = X.transpose(1,0,2).reshape(N,-1)
    corr = np.corrcoef(sensor_series)
    corr = np.nan_to_num(corr, nan=0.0)
    if gcfg.get('use_absolute_corr', True): corr = np.abs(corr)
    np.fill_diagonal(corr, 1.0)

    topo_file = gcfg.get('topology_file')
    if topo_file and Path(topo_file).exists():
        topo = np.loadtxt(topo_file, delimiter=',')
    else:
        topo = np.eye(N, dtype='float32')
    alpha = float(gcfg.get('alpha',1.0))
    A = alpha*corr + (1-alpha)*topo

    top_k = gcfg.get('top_k', None)
    threshold = float(gcfg.get('threshold',0.0))
    mask = np.zeros_like(A, dtype=bool)
    for i in range(N):
        idx = np.argsort(A[i])[-int(top_k):] if top_k else np.arange(N)
        mask[i, idx] = True
    A = np.where((A >= threshold) & (mask | mask.T), A, 0.0)
    np.fill_diagonal(A, 1.0)
    A = ((A + A.T) / 2).astype('float32')
    return A


def adjacency_to_edge_index(A):
    rows, cols = np.nonzero(A)
    edge_index = torch.tensor(np.vstack([rows, cols]), dtype=torch.long)
    edge_weight = torch.tensor(A[rows, cols], dtype=torch.float32)
    return edge_index, edge_weight
