# Deep Learning Architectures

A technical research repository implementing, analyzing, and customizing deep learning architectures across computer vision, sequence modeling, natural language processing, time series, graphs, generative AI, and reinforcement learning.

## Research Philosophy

**Business Problem → Data Characteristics → Architecture → Experiment → Optimization → Ablation → Error Analysis → Business Value**

The objective is to demonstrate not only how to use deep learning frameworks, but how architectural choices affect model behavior, computational cost, generalization, and real-world performance.

## Technology Stack

| Area | Technologies |
|---|---|
| Programming | Python |
| Numerical Computing | NumPy, SciPy |
| Data | Pandas |
| Deep Learning | PyTorch, TensorFlow, Keras |
| Computer Vision | TorchVision, OpenCV |
| NLP / Transformers | Hugging Face Transformers |
| Graph Learning | PyTorch Geometric |
| Reinforcement Learning | Gymnasium, Stable-Baselines3 |
| Experiment Tracking | MLflow, Weights & Biases |
| GPU | CUDA, cuDNN |
| Visualization | Matplotlib, Plotly |
| Development | Jupyter, Google Colab |
| Version Control | Git, GitHub |

## Repository Structure

```text
deep-learning-architectures/
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── fundamentals/
│   ├── perceptron/
│   ├── neural_network/
│   ├── forward_propagation/
│   └── backpropagation/
├── cnn/
│   ├── lenet/
│   ├── alexnet/
│   ├── vgg/
│   ├── inception/
│   ├── resnet/
│   ├── densenet/
│   ├── mobilenet/
│   └── efficientnet/
├── cnn_customization/
│   ├── dilated_convolution/
│   ├── multi_scale/
│   ├── deformable_convolution/
│   ├── attention_cnn/
│   └── three_dimensional_cnn/
├── rnn/
│   ├── vanilla_rnn/
│   ├── lstm/
│   ├── gru/
│   └── bidirectional_rnn/
├── transformers/
│   ├── attention/
│   ├── encoder/
│   ├── decoder/
│   ├── encoder_decoder/
│   └── time_series_transformer/
├── autoencoders/
│   ├── autoencoder/
│   ├── denoising/
│   ├── sparse/
│   └── variational/
├── generative/
│   ├── gan/
│   ├── dcgan/
│   ├── conditional_gan/
│   └── diffusion/
├── gnn/
│   ├── gcn/
│   ├── graphsage/
│   └── gat/
├── reinforcement_learning/
│   ├── q_learning/
│   ├── dqn/
│   └── ppo/
├── experiments/
├── ablation_studies/
├── error_analysis/
├── notebooks/
├── src/
├── tests/
└── docs/
```

## Architecture Families

### Neural Network Foundations

- Perceptron
- Multilayer Perceptron
- Forward propagation
- Backpropagation
- Loss functions
- Gradient descent
- Activation functions
- Weight initialization
- Batch normalization
- Layer normalization
- Dropout

### Convolutional Neural Networks

- LeNet
- AlexNet
- VGG
- Inception / GoogLeNet
- ResNet
- DenseNet
- MobileNet
- EfficientNet

### CNN Research and Customization

- Dilated convolution
- Multi-scale convolution
- Deformable convolution
- 3D convolution
- Attention-enhanced CNNs
- Residual architecture customization
- Feature pyramid designs

### Sequence Models

- Vanilla RNN
- Bidirectional RNN
- LSTM
- GRU
- CNN-LSTM
- Temporal convolutional networks

### Transformers

- Scaled dot-product attention
- Self-attention
- Multi-head attention
- Positional encoding
- Encoder
- Decoder
- Encoder-decoder
- Vision Transformer
- Time-series Transformer
- Transformer-based NLP

### Autoencoders

- Autoencoder
- Denoising Autoencoder
- Sparse Autoencoder
- Variational Autoencoder

### Generative Models

- GAN
- DCGAN
- Conditional GAN
- Diffusion models

### Graph Neural Networks

- GCN
- GraphSAGE
- GAT

### Reinforcement Learning

- Q-learning
- DQN
- Policy gradients
- Actor-critic
- PPO

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

Architecture selection should be driven by the characteristics of the problem and data.

| Data / Problem | Candidate Architectures |
|---|---|
| Tabular | MLP |
| Images | CNN, ResNet, ViT |
| Video | 3D CNN, CNN-LSTM, Video Transformer |
| Text | RNN, LSTM, Transformer |
| Time Series | LSTM, GRU, TCN, Transformer |
| Graph Data | GCN, GraphSAGE, GAT |
| Anomaly Detection | Autoencoder, VAE |
| Generation | GAN, VAE, Diffusion |
| Sequential Decisions | DQN, PPO, Actor-Critic |
| Multimodal Data | Fusion architectures, Multimodal Transformers |

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

## Example Research Question

**Does multi-scale dilated convolution improve detection of small and large features compared with a standard residual CNN under the same training budget?**

A research experiment can compare:

1. Baseline ResNet
2. Dilated ResNet
3. Multi-scale ResNet
4. Attention-enhanced ResNet
5. Ablation of dilation
6. Ablation of attention
7. Computational cost
8. Accuracy and error characteristics

## Industry Applications

- **Finance:** sequence forecasting, fraud detection, risk modeling
- **Aerospace:** sensor fusion, predictive maintenance, image inspection
- **Space:** satellite imagery, telemetry anomaly detection, remote sensing
- **Oil & Gas:** seismic interpretation, equipment monitoring, production forecasting
- **Energy:** load forecasting, renewable forecasting, battery health
- **Supply Chain:** demand forecasting, computer vision, anomaly detection

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

🚧 Active research and portfolio development.
