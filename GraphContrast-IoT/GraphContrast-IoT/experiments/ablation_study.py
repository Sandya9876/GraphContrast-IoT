from copy import deepcopy
from src.utils import load_config, save_json
from main import run_once

base = load_config('configs/skab.yaml')
settings = {
    'full': {},
    'no_sensor_dropout': {'augmentation.sensor_dropout': 0.0},
    'no_noise': {'augmentation.noise_std': 0.0},
    'no_edge_dropout': {'augmentation.edge_dropout': 0.0},
    'correlation_only': {'graph.alpha': 1.0},
}

def set_nested(cfg, dotted, value):
    parts=dotted.split('.'); cur=cfg
    for p in parts[:-1]: cur=cur[p]
    cur[parts[-1]]=value

results={}
for name, changes in settings.items():
    cfg=deepcopy(base)
    cfg['training']['seeds']=[42]
    for k,v in changes.items(): set_nested(cfg,k,v)
    *_, metrics, _ = run_once(cfg, 42)
    results[name]=metrics
save_json(results, 'results/metrics/ablation_skab.json')
print(results)
