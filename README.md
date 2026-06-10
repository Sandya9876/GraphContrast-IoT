
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20625406.svg)](https://doi.org/10.5281/zenodo.20625406)


# GraphContrast-IoT

## GraphContrast-IoT: A Contrastive Self-Supervised GCT-ADNet Framework for Robust Time-Series Anomaly Detection in IoT Sensor Data

### Overview

GraphContrast-IoT is a novel self-supervised graph neural network framework for anomaly detection in multivariate IoT sensor networks. The framework integrates hybrid graph construction, graph neural networks, temporal sequence modeling, and contrastive representation learning to detect anomalies without requiring labeled attack or fault data.

Unlike traditional supervised approaches, GraphContrast-IoT learns robust spatio-temporal sensor representations directly from normal operational data, making it suitable for real-world industrial IoT deployments where anomaly labels are scarce, expensive, or unavailable.

---

## Key Features

* Hybrid Topology-Aware Graph Construction
* Graph Neural Networks (GCN/GAT)
* Temporal Sequence Modeling (GRU/TCN)
* Multi-View Contrastive Learning
* Self-Supervised Representation Learning
* Embedding-Based Anomaly Detection
* End-to-End Training Pipeline
* Support for SWaT, WADI, and SKAB Datasets
* GPU Acceleration with PyTorch Geometric
* Reproducible Research Framework

---

## Framework Architecture

```text
Raw IoT Sensor Streams
          │
          ▼
Preprocessing & Windowing
          │
          ▼
Hybrid Graph Construction
 (Topology + Correlation)
          │
          ▼
Multi-View Augmentation
          │
          ▼
GCT-ADNet Encoder
 ├── Graph Convolution
 ├── Temporal Encoder
 └── Feature Fusion
          │
          ▼
Projection Head
          │
          ▼
Contrastive Learning
      (InfoNCE)
          │
          ▼
Latent Embeddings
          │
          ▼
Anomaly Scoring
          │
          ▼
Detected Anomalies
```

---

## Supported Datasets

| Dataset | Domain                        | Description                                  |
| ------- | ----------------------------- | -------------------------------------------- |
| SWaT    | Industrial Control Systems    | Water treatment cyber-physical system        |
| WADI    | Water Distribution Networks   | Industrial water distribution infrastructure |
| SKAB    | IoT Water Circulation Systems | Benchmark dataset for anomaly detection      |

---

## Installation

### Clone Repository

```bash
git clone https://github.com/Sandya9876/GraphContrast-IoT

cd GraphContrast-IoT
```

### Create Environment

```bash
conda env create -f environment.yml

conda activate graphcontrast-iot
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Requirements

* Python 3.10+
* PyTorch 2.2+
* PyTorch Geometric 2.5+
* NumPy
* Pandas
* SciPy
* Scikit-learn
* NetworkX
* Matplotlib
* Seaborn
* PyYAML
* TQDM

---

## Training

### SKAB

```bash
python train.py --dataset skab
```

### SWaT

```bash
python train.py --dataset swat
```

### WADI

```bash
python train.py --dataset wadi
```

---

## Inference

```bash
python detect.py \
--dataset skab \
--checkpoint models/checkpoints/best_model.pt
```

---

## Evaluation

```bash
python evaluate.py --dataset skab
```

---

## Methodology

### Hybrid Graph Construction

The sensor graph is constructed by combining statistical correlations and physical topology information:

```math
A = αA_corr + (1-α)A_topology
```

### Contrastive Learning

GraphContrast-IoT employs an InfoNCE-based contrastive objective to learn robust latent representations from multiple augmented graph views.

### Anomaly Detection

Anomaly scores are computed using distances from the learned normal embedding centroid:

```math
s_i = ||z_i - c||_2
```

Samples exceeding a learned threshold are identified as anomalies.

---

## Generated Outputs

The framework automatically generates:

* Training Loss Curve
* ROC Curve
* Precision-Recall Curve
* Confusion Matrix
* Anomaly Score Timeline
* Performance Metrics
* Model Checkpoints

---

## Ablation Studies

Implemented ablation experiments include:

1. Without Graph Module
2. Without Contrastive Learning
3. Without Temporal Encoder
4. Without Sensor Dropout
5. Without Edge Perturbation
6. Correlation-Only Graph
7. Topology-Only Graph
8. Full Model

```bash
python experiments/ablation_study.py
```

---

## Applications

* Industrial IoT Monitoring
* Smart Manufacturing
* Critical Infrastructure Protection
* Cyber-Physical Security

---

## Citation

```bibtex
@article{GraphContrastIoT2026,
  title={GraphContrast-IoT: A Contrastive Self-Supervised GCT-ADNet Framework for Robust Time-Series Anomaly Detection in IoT Sensor Data},
  year={2026}
}
```

---

## License

This project is released under the MIT License.

---



