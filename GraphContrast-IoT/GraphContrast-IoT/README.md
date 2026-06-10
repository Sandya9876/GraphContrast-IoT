# GraphContrast-IoT

Complete PyTorch/PyTorch-Geometric implementation of **GraphContrast-IoT: A Contrastive Self-Supervised GCT-ADNet Framework for Robust Time-Series Anomaly Detection in IoT Sensor Data**.

## Quick start

```bash
pip install -r requirements.txt
python main.py --config configs/skab.yaml
```

Place dataset CSV files in:

```text
data/raw/SKAB/
data/raw/SWaT/
data/raw/WADI/
```

The code automatically detects numeric sensor columns and common label columns such as `label`, `attack`, `anomaly`, `Normal/Attack`, or `class`. If no CSV is available, it can generate a small synthetic IoT dataset for pipeline testing when `allow_synthetic: true` in the config.

## Outputs

Results are saved to:

```text
results/metrics/
results/figures/
results/anomaly_scores/
models/checkpoints/
```
