# 50 Deep Learning Algorithms — Full Architecture Coverage

This repository is a structured deep-learning architecture reference and implementation portfolio. It covers foundational neural networks, convolutional models, recurrent and spiking networks, autoencoders, generative models, attention/transformers, graph neural networks, deep reinforcement learning, and specialized architectures.

## Architecture Coverage Matrix

| # | Algorithm / Architecture | Family | Primary Data | Core Purpose |
|---:|---|---|---|---|
| 1 | Perceptron | Neural foundation | Tabular/vector | Linear classification |
| 2 | Multilayer Perceptron (MLP/ANN) | Neural foundation | Tabular/vector | Nonlinear classification/regression |
| 3 | Sigmoid Neural Network | Activation/output | Scalar/vector | Binary probability and gating |
| 4 | Radial Basis Function Network (RBF) | Kernel/neural | Tabular/vector | Nonlinear approximation/classification |
| 5 | Convolutional Neural Network (CNN) | Vision | Image/grid | Spatial feature extraction |
| 6 | LeNet-5 | CNN | Images | Digit/image classification |
| 7 | AlexNet | CNN | Images | Large-scale image classification |
| 8 | VGG | CNN | Images | Deep stacked convolution |
| 9 | Inception / GoogLeNet | CNN | Images | Multi-scale feature extraction |
| 10 | ResNet | CNN | Images | Residual deep networks |
| 11 | DenseNet | CNN | Images | Dense feature reuse |
| 12 | MobileNet | CNN | Images/mobile | Efficient depthwise convolution |
| 13 | EfficientNet | CNN | Images | Compound model scaling |
| 14 | Vision Transformer (ViT) | Transformer | Images | Patch-based attention |
| 15 | Vanilla RNN | Recurrent | Sequences | Temporal modeling |
| 16 | LSTM | Recurrent | Sequences/time series | Long-term dependencies |
| 17 | GRU | Recurrent | Sequences/time series | Gated temporal modeling |
| 18 | Bidirectional RNN | Recurrent | Sequences | Forward/backward context |
| 19 | CNN-LSTM | Hybrid | Video/time series | Spatial + temporal modeling |
| 20 | Temporal Convolutional Network (TCN) | Temporal CNN | Time series | Causal temporal modeling |
| 21 | Liquid State Machine (LSM) | Spiking | Event streams | Neuromorphic temporal processing |
| 22 | Autoencoder (AE) | Representation | Vector/image | Reconstruction/feature learning |
| 23 | Denoising Autoencoder (DAE) | Autoencoder | Noisy data | Robust representation |
| 24 | Sparse Autoencoder (SAE) | Autoencoder | Vector/image | Sparse feature learning |
| 25 | Variational Autoencoder (VAE) | Generative | Vector/image | Probabilistic latent representation |
| 26 | Conditional VAE (CVAE) | Generative | Labeled data | Conditional generation |
| 27 | Generative Adversarial Network (GAN) | Generative | Images/data | Adversarial generation |
| 28 | Deep Convolutional GAN (DCGAN) | GAN | Images | Convolutional generation |
| 29 | Conditional GAN (cGAN) | GAN | Labeled data | Class-conditioned generation |
| 30 | Wasserstein GAN (WGAN) | GAN | Images/data | Improved adversarial optimization |
| 31 | Diffusion Model | Generative | Images/data | Iterative denoising generation |
| 32 | U-Net | Encoder-decoder | Images | Segmentation/dense prediction |
| 33 | Attention Mechanism | Attention | Sequences/images | Context weighting |
| 34 | Transformer Encoder | Transformer | Text/sequences | Parallel contextual encoding |
| 35 | Transformer Decoder | Transformer | Text/sequences | Autoregressive generation |
| 36 | Encoder-Decoder Transformer | Transformer | Sequence-to-sequence | Translation/generation |
| 37 | BERT-style Transformer | Transformer/NLP | Text | Bidirectional language representation |
| 38 | GPT-style Transformer | Transformer/NLP | Text | Autoregressive language modeling |
| 39 | Graph Convolutional Network (GCN) | GNN | Graphs | Graph representation learning |
| 40 | GraphSAGE | GNN | Graphs | Inductive node embeddings |
| 41 | Graph Attention Network (GAT) | GNN | Graphs | Attention over graph neighbors |
| 42 | Graph Autoencoder (GAE) | GNN/generative | Graphs | Graph embedding/reconstruction |
| 43 | Deep Q-Network (DQN) | Deep RL | States/actions | Value-based reinforcement learning |
| 44 | Double DQN | Deep RL | States/actions | Reduced Q-value overestimation |
| 45 | Dueling DQN | Deep RL | States/actions | Separate value/advantage estimation |
| 46 | Deep Deterministic Policy Gradient (DDPG) | Deep RL | Continuous control | Actor-critic control |
| 47 | Proximal Policy Optimization (PPO) | Deep RL | Sequential decisions | Stable policy optimization |
| 48 | Deep Belief Network (DBN) | Probabilistic deep learning | Vector/data | Layered generative representation |
| 49 | Deep Boltzmann Machine (DBM) | Energy-based | Vector/data | Undirected latent representation |
| 50 | Capsule Network (CapsNet) | Vision | Images | Part-whole representation |

## Full Architecture Documentation Standard

Every algorithm should eventually contain the following artifacts:

1. **Overview** — definition, purpose, history, and problem class.
2. **Architecture** — complete layer/block topology.
3. **Mathematics** — equations, transformations, activation functions, and objective function.
4. **Data Flow** — input → preprocessing → model → output.
5. **Tensor Shapes** — dimensions at every major layer.
6. **Parameters** — trainable parameter calculations and model size.
7. **Forward Pass** — step-by-step computation.
8. **Backward Pass** — gradient flow and optimization.
9. **Loss Function** — mathematical definition and implementation.
10. **Optimizer** — SGD, Adam, AdamW, RMSprop, or architecture-specific method.
11. **Regularization** — dropout, weight decay, normalization, sparsity, augmentation, or other controls.
12. **Hyperparameters** — learning rate, batch size, depth, width, sequence length, latent dimension, etc.
13. **Training Pipeline** — dataset, split, preprocessing, training loop, validation, checkpointing.
14. **Evaluation** — appropriate metrics and baseline comparison.
15. **Error Analysis** — failure modes and misclassification patterns.
16. **Ablation Study** — isolate the effect of major architectural components.
17. **Computational Profile** — parameters, FLOPs/MACs, memory, latency, and scalability.
18. **Implementation** — runnable Python implementation.
19. **Framework Versions** — PyTorch and/or TensorFlow/Keras implementation where appropriate.
20. **Applications** — finance, aerospace, space, oil & gas, energy, supply chain, cybersecurity, NLP, and computer vision.
21. **Customization** — methods for modifying the architecture for a domain-specific problem.
22. **Experiment Notebook** — reproducible experiment with results.
23. **Architecture Diagram** — visual representation of the network.
24. **Testing** — unit tests for data flow, dimensions, loss, and inference.
25. **References** — original papers and authoritative technical sources.

## Standard Directory Structure

```text
algorithms/
├── 01_perceptron/
├── 02_mlp_ann/
├── 03_sigmoid/
├── 04_rbf/
├── 05_cnn/
├── 06_lenet/
├── 07_alexnet/
├── 08_vgg/
├── 09_inception/
├── 10_resnet/
├── 11_densenet/
├── 12_mobilenet/
├── 13_efficientnet/
├── 14_vit/
├── 15_rnn/
├── 16_lstm/
├── 17_gru/
├── 18_bidirectional_rnn/
├── 19_cnn_lstm/
├── 20_tcn/
├── 21_lsm/
├── 22_autoencoder/
├── 23_denoising_autoencoder/
├── 24_sparse_autoencoder/
├── 25_vae/
├── 26_cvae/
├── 27_gan/
├── 28_dcgan/
├── 29_cgan/
├── 30_wgan/
├── 31_diffusion/
├── 32_unet/
├── 33_attention/
├── 34_transformer_encoder/
├── 35_transformer_decoder/
├── 36_encoder_decoder_transformer/
├── 37_bert/
├── 38_gpt/
├── 39_gcn/
├── 40_graphsage/
├── 41_gat/
├── 42_graph_autoencoder/
├── 43_dqn/
├── 44_double_dqn/
├── 45_dueling_dqn/
├── 46_ddpg/
├── 47_ppo/
├── 48_dbn/
├── 49_dbm/
└── 50_capsnet/
```

## Recommended File Pattern

Each algorithm directory should use:

```text
algorithm_name/
├── README.md
├── model.py
├── train.py
├── evaluate.py
├── inference.py
├── config.yaml
├── requirements.txt
├── tests/
│   └── test_model.py
├── notebooks/
│   └── experiment.ipynb
├── diagrams/
│   └── architecture.png
└── results/
    └── metrics.json
```

## Cross-Architecture Comparison

### Data modality

- Tabular: MLP, RBF, AE, VAE
- Images: CNN families, ViT, U-Net, GANs, diffusion, CapsNet
- Time series: RNN, LSTM, GRU, TCN, Transformer
- Text: RNN/LSTM/GRU, Transformer, BERT, GPT
- Graphs: GCN, GraphSAGE, GAT, GAE
- Event streams: LSM
- Sequential decisions: DQN family, DDPG, PPO

### Learning objective

- Classification
- Regression
- Sequence prediction
- Representation learning
- Dimensionality reduction
- Anomaly detection
- Image segmentation
- Generation
- Language modeling
- Graph prediction
- Reinforcement learning
- Continuous control

## Portfolio Research Workflow

```text
Business / Engineering Problem
          ↓
Data Characterization
          ↓
Baseline Model
          ↓
Architecture Selection
          ↓
Model Implementation
          ↓
Training
          ↓
Hyperparameter Optimization
          ↓
Ablation Study
          ↓
Error Analysis
          ↓
Performance + Computational Evaluation
          ↓
Domain Validation
          ↓
Deployment / Reproducibility
```

## Industry Mapping

| Industry | Example architectures |
|---|---|
| Finance | LSTM, GRU, TCN, Transformer, GNN, VAE |
| Aerospace | CNN, 3D CNN, LSTM, Transformer, GNN |
| Space | CNN, ViT, U-Net, autoencoder, diffusion |
| Oil & Gas | CNN, LSTM, GRU, TCN, VAE |
| Energy | LSTM, GRU, TCN, Transformer |
| Supply Chain | MLP, LSTM, GRU, Transformer, GNN |
| Cybersecurity | Autoencoder, VAE, CNN, RNN, GNN |
| NLP | Transformer, BERT, GPT, RNN |
| Computer Vision | CNN, ResNet, ViT, U-Net, GAN, diffusion |
| Robotics | CNN, LSTM, GNN, DQN, DDPG, PPO |

## Implementation Priority

The repository should progress from foundational models to increasingly specialized architectures:

**Level 1 — Foundations:** 1–5  
**Level 2 — CNNs:** 6–14  
**Level 3 — Sequence Models:** 15–21  
**Level 4 — Representation Learning:** 22–26  
**Level 5 — Generative Models:** 27–32  
**Level 6 — Attention & Transformers:** 33–38  
**Level 7 — Graph Learning:** 39–42  
**Level 8 — Deep Reinforcement Learning:** 43–47  
**Level 9 — Specialized Architectures:** 48–50

## Author

**Whitney Moss**

Machine Learning Engineering | Deep Learning | Applied Analytics | Quantitative Analytics | Systems Engineering

## Status

Active research and portfolio development.
