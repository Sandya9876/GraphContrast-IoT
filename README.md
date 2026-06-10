GraphContrast-IoT
GraphContrast-IoT: A Contrastive Self-Supervised GCT-ADNet Framework for Robust Time-Series Anomaly Detection in IoT Sensor Data
________________________________________
Overview
GraphContrast-IoT is a self-supervised graph neural network framework designed for anomaly detection in multivariate IoT sensor networks. The framework combines:
	Hybrid Graph Construction 
	Graph Neural Networks (GCN/GAT) 
	Temporal Sequence Modeling (GRU/TCN) 
	Contrastive Self-Supervised Learning 
	Embedding-Based Anomaly Detection 
Unlike traditional supervised approaches, GraphContrast-IoT learns meaningful sensor representations without requiring anomaly labels during training, making it suitable for real-world industrial IoT deployments where anomalies are rare and expensive to annotate.
________________________________________
Key Features
✅ Hybrid topology-aware graph construction
✅ Multi-view contrastive learning
✅ Graph Convolution + Temporal Encoding
✅ End-to-end self-supervised training
✅ Embedding-based anomaly scoring
✅ Supports SWaT, WADI, and SKAB datasets
✅ Reproducible experiments
✅ GPU acceleration using PyTorch Geometric
✅ Modular research-oriented implementation
________________________________________
Framework Architecture
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
________________________________________
Repository Structure
GraphContrast-IoT/
│
├── README.md
├── requirements.txt
├── environment.yml
│
├── main.py
├── train.py
├── detect.py
├── evaluate.py
│
├── configs/
│   ├── swat.yaml
│   ├── wadi.yaml
│   └── skab.yaml
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── splits/
│
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── windowing.py
│   ├── graph_builder.py
│   ├── augmentations.py
│   ├── model.py
│   ├── losses.py
│   ├── metrics.py
│   ├── visualization.py
│   └── utils.py
│
├── experiments/
│   ├── run_swat.py
│   ├── run_wadi.py
│   ├── run_skab.py
│   ├── ablation_study.py
│   └── cross_dataset_eval.py
│
├── models/
│   ├── checkpoints/
│   └── saved_encoders/
│
├── results/
│   ├── metrics/
│   ├── figures/
│   ├── anomaly_scores/
│   └── logs/
│
└── notebooks/
________________________________________
Supported Datasets
1. SWaT
Secure Water Treatment Dataset
Domain:
Industrial Control System

Sensors:
51

Anomaly Types:
Cyber-attacks
Process attacks
Physical attacks
Dataset Request:
https://itrust.sutd.edu.sg/itrust-labs_datasets/dataset_info/
________________________________________
2. WADI
Water Distribution Dataset
Domain:
Industrial Water Distribution Network

Sensors:
123+

Anomaly Types:
Cyber-attacks
Faults
Abnormal operations
Dataset Request:
https://itrust.sutd.edu.sg/itrust-labs_datasets/
________________________________________
3. SKAB
Skoltech Anomaly Benchmark
Domain:
IoT Water Circulation System

Anomaly Types:
Point anomalies
Collective anomalies
Repository:
https://github.com/waico/SKAB
________________________________________
Installation
Clone Repository
git clone https://github.com/yourusername/GraphContrast-IoT.git

cd GraphContrast-IoT
________________________________________
Create Environment
Using Conda:
conda env create -f environment.yml

conda activate graphcontrast-iot
________________________________________
Install Dependencies
pip install -r requirements.txt
________________________________________
Requirements
Python 3.10+

PyTorch 2.2+

PyTorch Geometric 2.5+

NumPy
Pandas
SciPy
Scikit-learn
NetworkX
Matplotlib
Seaborn
PyYAML
TQDM
________________________________________
Dataset Preparation
Create folders:
data/
└── raw/
    ├── SWaT/
    ├── WADI/
    └── SKAB/
Place downloaded dataset files in the corresponding directories.
Example:
data/raw/SKAB/
    anomaly-free.csv
    anomaly.csv
________________________________________
Training
SKAB
python train.py --dataset skab
SWaT
python train.py --dataset swat
WADI
python train.py --dataset wadi
________________________________________
Inference
python detect.py \
--dataset skab \
--checkpoint models/checkpoints/best_model.pt
________________________________________
Evaluation
python evaluate.py \
--dataset skab
________________________________________
Hyperparameters
Default configuration:
window_size: 100

stride: 10

hidden_dim: 128

embedding_dim: 64

gcn_layers: 2

dropout: 0.3

batch_size: 64

epochs: 100

learning_rate: 0.001

temperature: 0.2

alpha: 0.5
________________________________________
Multi-View Augmentations
Implemented augmentations:
Sensor Dropout
Random sensor masking.
Magnitude Perturbation
Gaussian noise injection.
Temporal Warping
Temporal interpolation and stretching.
Edge Perturbation
Random graph edge modification.
________________________________________
Graph Construction
Hybrid graph formulation:
A=αA_corr+(1-α)A_topology

Where:
A_corr:
Correlation matrix

A_topology:
Physical connectivity matrix

α:
Fusion parameter
________________________________________
Contrastive Learning
InfoNCE objective:
L=-log (exp⁡(sim(z_i,z_j)/τ))/(∑_k▒〖exp⁡(sim(〗 z_i,z_k)/τ))

Where:
z_i, z_j
Positive pair embeddings

τ
Temperature parameter
________________________________________
Anomaly Detection
Normal centroid:
c=1/N ∑_i▒z_i 

Anomaly score:
s_i=∣∣z_i-c∣∣_2

Threshold:
T=μ+kσ

Decision rule:
s_i>T→anomaly

________________________________________
Output Files
Training produces:
models/checkpoints/
    best_model.pt

results/
├── metrics/
├── figures/
├── logs/
└── anomaly_scores/
________________________________________
Generated Figures
The framework automatically generates:
Training Loss Curve
loss_curve.png
ROC Curve
roc_curve.png
Precision-Recall Curve
pr_curve.png
Confusion Matrix
confusion_matrix.png
Anomaly Score Timeline
anomaly_scores.png
________________________________________
Reproducibility
Experiments are executed using:
Random Seeds:
42
123
2024
Results reported in the manuscript correspond to the average across three independent runs.
________________________________________
Ablation Studies
Implemented ablations:
1. Without Graph Module

2. Without Contrastive Learning

3. Without Temporal Encoder

4. Without Sensor Dropout

5. Without Edge Perturbation

6. Correlation-only Graph

7. Topology-only Graph

8. Full Model
Run:
python experiments/ablation_study.py
________________________________________
Computational Complexity
Graph Module:
O(L(∣E∣d+Nd^2))

Temporal Encoder:
O(NWd^2)

Contrastive Learning:
O(B^2 d)

Where:
N = nodes
E = edges
W = window size
B = batch size
d = hidden dimension
________________________________________
Citation
@article{GraphContrastIoT2026,
  title={GraphContrast-IoT: A Contrastive Self-Supervised GCT-ADNet Framework for Robust Time-Series Anomaly Detection in IoT Sensor Data},
  author={Author Name},
  journal={Under Review},
  year={2026}
}
________________________________________
License
This project is released under the MIT License.
________________________________________
Contact
For research collaborations, implementation questions, or reproducibility issues:
Author: <Your Name>

Email: <your_email>

Institution: <your_institution>

