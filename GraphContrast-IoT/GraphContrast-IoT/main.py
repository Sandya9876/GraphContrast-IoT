import argparse
from pathlib import Path
import numpy as np
import torch
from src.utils import load_config, set_seed, get_device, ensure_dirs, save_json
from src.data_loader import load_dataset
from src.graph_builder import build_adjacency, adjacency_to_edge_index
from src.model import GCTADNet
from src.visualization import plot_loss, plot_scores, plot_roc_pr, plot_cm
from train import train_model
from detect import embed, fit_centroid_threshold, score_embeddings
from evaluate import evaluate_predictions


def run_once(cfg, seed):
    set_seed(seed)
    device = get_device(cfg['training'].get('device','auto'))
    data = load_dataset(cfg, seed)
    Xtr,ytr = data['train']; Xv,yv = data['val']; Xte,yte = data['test']
    A = build_adjacency(Xtr, cfg)
    edge_index, edge_weight = adjacency_to_edge_index(A)
    N,W,F = Xtr.shape[1], Xtr.shape[2], Xtr.shape[3]
    mcfg = cfg['model']
    model = GCTADNet(N,W,F,mcfg['hidden_dim'],mcfg['embedding_dim'],mcfg['gnn_layers'],mcfg['graph_type'],mcfg['temporal_type'],mcfg['dropout']).to(device)
    model, losses = train_model(model, Xtr, Xv, edge_index, edge_weight, cfg, device)
    bs = cfg['training']['batch_size']
    ztr = embed(model, Xtr, edge_index, edge_weight, bs, device)
    zv = embed(model, Xv, edge_index, edge_weight, bs, device)
    zte = embed(model, Xte, edge_index, edge_weight, bs, device)
    normal_train = ztr[ytr==0] if (ytr==0).any() else ztr
    centroid, threshold, val_scores = fit_centroid_threshold(normal_train, zv, yv, cfg['detection']['threshold_k'])
    scores, pred = score_embeddings(zte, centroid, threshold)
    metrics = evaluate_predictions(yte, scores, pred)
    metrics['seed'] = seed; metrics['threshold'] = threshold
    return model, losses, scores, pred, yte, metrics, A


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/skab.yaml')
    args = parser.parse_args()
    cfg = load_config(args.config)
    name = cfg['dataset']['name']
    ensure_dirs('results/metrics','results/figures','results/anomaly_scores','models/checkpoints')
    all_metrics=[]
    last = None
    for seed in cfg['training'].get('seeds',[42]):
        model, losses, scores, pred, yte, metrics, A = run_once(cfg, seed)
        all_metrics.append(metrics); last=(model, losses, scores, pred, yte, A, seed)
        torch.save(model.state_dict(), f'models/checkpoints/{name}_seed{seed}.pt')
        np.save(f'results/anomaly_scores/{name}_scores_seed{seed}.npy', scores)
    avg = {}
    for k in all_metrics[0].keys():
        vals=[m[k] for m in all_metrics if isinstance(m[k], (int,float)) and m[k] is not None]
        if vals: avg[k+'_mean'] = float(np.mean(vals)); avg[k+'_std'] = float(np.std(vals))
    save_json({'runs': all_metrics, 'average': avg}, f'results/metrics/{name}_metrics.json')
    model, losses, scores, pred, yte, A, seed = last
    plot_loss(losses, f'results/figures/{name}_loss.png')
    plot_scores(scores, all_metrics[-1]['threshold'], yte, f'results/figures/{name}_scores.png')
    plot_roc_pr(yte, scores, 'results/figures', name)
    plot_cm(yte, pred, f'results/figures/{name}_cm.png')
    np.save(f'results/anomaly_scores/{name}_adjacency.npy', A)
    print('Completed:', name)
    print(avg)

if __name__ == '__main__':
    main()
