# Deep Learning Architectures

A technical research repository implementing, analyzing, and customizing deep learning architectures across computer vision, sequence modeling, generative AI, time series, graphs, neuromorphic computing, and reinforcement learning.

## Chat Project: Deep Learning Algorithms

This repository now includes the deep-learning algorithms discussed in this project:

- Artificial Neural Network (ANN)
- Sigmoid activation
- Convolutional Neural Network (CNN)
- Recurrent Neural Network (RNN)
- Long Short-Term Memory (LSTM)
- Gated Recurrent Unit (GRU)
- Liquid State Machine (LSM)
- Radial Basis Function Network (RBF)
- Sparse Autoencoder (SAE)
- Variational Autoencoder (VAE)
- Generative Adversarial Network (GAN)

## Repository Structure

```text
deep-learning-architectures/
├── README.md
├── requirements.txt
├── algorithms/
│   ├── ann_sigmoid.py
│   ├── cnn.py
│   ├── lstm.py
│   ├── gru.py
│   ├── rbf.py
│   ├── sparse_autoencoder.py
│   ├── vae.py
│   └── lsm.py
├── fundamentals/
├── cnn/
├── rnn/
├── transformers/
├── autoencoders/
├── generative/
├── gnn/
├── reinforcement_learning/
├── experiments/
├── ablation_studies/
├── error_analysis/
├── notebooks/
├── src/
├── tests/
└── docs/
```

## Algorithm Catalog

| Algorithm | Data | Typical use |
|---|---|---|
| ANN | Tabular/vector | Classification and regression |
| Sigmoid | Scalar/vector | Binary output probability and gating |
| CNN | Images/spatial signals | Vision and spatial feature extraction |
| RNN | Sequential data | Sequence and time-series modeling |
| LSTM | Sequential data | Long-range temporal dependencies |
| GRU | Sequential data | Efficient temporal modeling |
| LSM | Event streams/sequences | Spiking and neuromorphic computing |
| RBF Network | Vector/tabular | Classification and function approximation |
| SAE | Vector/tabular | Sparse representation learning |
| VAE | Vector/image | Probabilistic representation and generation |
| GAN | Images/other data | Generative modeling |

## Model Research Standard

Each substantial architecture will document:

1. Historical development
2. Problem addressed
3. Mathematical foundation
4. Architecture diagram
5. Layer-by-layer design
6. Input/output dimensions
7. Parameter count
8. Computational considerations
9. Training strategy
10. Hyperparameters
11. Baseline comparison
12. Performance metrics
13. Error analysis
14. Ablation study
15. Business applications
16. Industry applications
17. Customization opportunities

## Architecture Selection

| Data / Problem | Candidate architectures |
|---|---|
| Tabular | MLP / ANN |
| Images | CNN, ResNet, ViT |
| Video | 3D CNN, CNN-LSTM, Video Transformer |
| Text | RNN, LSTM, GRU, Transformer |
| Time Series | LSTM, GRU, TCN, Transformer |
| Event streams | LSM / spiking neural networks |
| Graph Data | GCN, GraphSAGE, GAT |
| Anomaly Detection | Autoencoder, SAE, VAE |
| Generation | GAN, VAE, Diffusion |

## Research Workflow

```text
Research Question
      ↓
Data Characterization
      ↓
Baseline Architecture
      ↓
Candidate Architecture
      ↓
Training
      ↓
Hyperparameter Optimization
      ↓
Ablation Study
      ↓
Error Analysis
      ↓
Model Comparison
      ↓
Business / Engineering Evaluation
```

## Industry Applications

- Finance: forecasting, fraud detection, risk modeling
- Aerospace: sensor fusion, predictive maintenance, image inspection
- Space: satellite imagery, telemetry anomaly detection, remote sensing
- Oil & Gas: seismic interpretation, equipment monitoring, production forecasting
- Energy: load forecasting, renewable forecasting, battery health
- Supply Chain: demand forecasting, computer vision, anomaly detection

## Evidence

Each research project should provide:

- Source code
- Notebook
- Dataset description
- Architecture diagram
- Training configuration
- Evaluation metrics
- Error analysis
- Experiment results
- Ablation results
- Technical documentation

## Author

**Whitney Moss**

Machine Learning Engineering | Deep Learning | Applied Analytics | Quantitative Analytics | Systems Engineering

## Status

Active research and portfolio development.
