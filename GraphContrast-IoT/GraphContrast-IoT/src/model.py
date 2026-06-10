import torch
from torch import nn
try:
    from torch_geometric.nn import GCNConv, GATConv
except Exception:
    GCNConv = GATConv = None

class DenseGCNLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__(); self.lin = nn.Linear(in_dim, out_dim)
    def forward(self, x, A_norm):
        # x [B,N,F], A_norm [N,N]
        return self.lin(torch.einsum('ij,bjf->bif', A_norm, x))

class GCTADNet(nn.Module):
    def __init__(self, num_sensors, window_size, input_dim=1, hidden_dim=128, embedding_dim=64,
                 gnn_layers=2, graph_type='gcn', temporal_type='gru', dropout=0.3):
        super().__init__()
        self.num_sensors = num_sensors; self.window_size = window_size
        self.hidden_dim = hidden_dim; self.graph_type = graph_type; self.temporal_type = temporal_type
        self.use_pyg = GCNConv is not None
        self.act = nn.ReLU(); self.drop = nn.Dropout(dropout)
        if self.use_pyg:
            convs = []; in_dim = input_dim
            for _ in range(gnn_layers):
                convs.append(GATConv(in_dim, hidden_dim, heads=1, dropout=dropout) if graph_type.lower() == 'gat' else GCNConv(in_dim, hidden_dim))
                in_dim = hidden_dim
            self.convs = nn.ModuleList(convs)
        else:
            layers=[]; in_dim=input_dim
            for _ in range(gnn_layers):
                layers.append(DenseGCNLayer(in_dim, hidden_dim)); in_dim=hidden_dim
            self.convs = nn.ModuleList(layers)
        if temporal_type.lower() == 'tcn':
            self.temporal = nn.Sequential(nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1), nn.ReLU(), nn.Dropout(dropout), nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1), nn.ReLU())
            temp_out = hidden_dim
        else:
            self.temporal = nn.GRU(hidden_dim, hidden_dim, batch_first=True); temp_out = hidden_dim
        self.fusion = nn.Sequential(nn.Linear(hidden_dim + temp_out, hidden_dim), nn.ReLU(), nn.Dropout(dropout))
        self.proj = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, embedding_dim))

    def _dense_adj(self, edge_index, edge_weight, N, device):
        A = torch.zeros(N,N,device=device)
        A[edge_index[0], edge_index[1]] = edge_weight if edge_weight is not None else 1.0
        A = A + torch.eye(N,device=device)
        deg = A.sum(1).clamp(min=1e-6)
        D = torch.diag(torch.pow(deg, -0.5))
        return D @ A @ D

    def forward(self, x, edge_index, edge_weight=None):
        B,N,W,F = x.shape
        if not self.use_pyg:
            A_norm = self._dense_adj(edge_index, edge_weight, N, x.device)
            hs=[]
            for t in range(W):
                h=x[:,:,t,:]
                for conv in self.convs:
                    h=self.drop(self.act(conv(h,A_norm)))
                hs.append(h)
            H=torch.stack(hs, dim=2)
        else:
            step_graph=[]
            for t in range(W):
                xt=x[:,:,t,:].reshape(B*N,F)
                eis=[]; ews=[]
                for b in range(B):
                    eis.append(edge_index + b*N)
                    if edge_weight is not None: ews.append(edge_weight)
                bei=torch.cat(eis, dim=1); bew=torch.cat(ews) if edge_weight is not None else None
                h=xt
                for conv in self.convs:
                    h = conv(h, bei) if self.graph_type.lower() == 'gat' else conv(h, bei, bew)
                    h=self.drop(self.act(h))
                step_graph.append(h.view(B,N,self.hidden_dim))
            H=torch.stack(step_graph, dim=2)
        spatial=H.mean(dim=2)
        seq=H.reshape(B*N,W,self.hidden_dim)
        if self.temporal_type.lower() == 'tcn':
            temp=self.temporal(seq.transpose(1,2)).mean(dim=2).view(B,N,self.hidden_dim)
        else:
            _,hn=self.temporal(seq); temp=hn[-1].view(B,N,self.hidden_dim)
        fused=self.fusion(torch.cat([spatial,temp],dim=-1)).mean(dim=1)
        z=self.proj(fused)
        return nn.functional.normalize(z, dim=1)
