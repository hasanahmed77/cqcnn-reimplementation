# CQ-CNN Reimplementation

## Overview

This repository contains a step-by-step reimplementation of a **hybrid classical–quantum convolutional neural network (CQ-CNN)** for Alzheimer’s disease classification using MRI data.

The goal of this project is to:

- Understand the architecture proposed in the original paper
- Rebuild the model from first principles
- Verify each component (CNN, QNN, and hybrid pipeline)
- Prepare for further research and extensions in quantum machine learning

---

## Architecture

The model follows a hybrid pipeline:

```
MRI Image (1 × 128 × 128)
        ↓
Convolutional Neural Network (CNN)
        ↓
Feature Compression (Fully Connected Layers)
        ↓
2-Dimensional Vector (for 2 qubits)
        ↓
Parameterized Quantum Circuit (PQC / QNN)
        ↓
Scalar Output
        ↓
Binary Classification Output
```

### Classical Component

- 2 convolutional layers
- Max pooling
- Dropout regularization
- Fully connected layers for dimensionality reduction

### Quantum Component

- **ZZFeatureMap** for data encoding
- **RealAmplitudes ansatz** (trainable quantum circuit)
- Implemented using **Qiskit Machine Learning**
- Integrated into PyTorch via `TorchConnector`

---

## Implemented Modules

### 1. Classical Backbone

`my_model_test.py`

- Reimplementation of CNN architecture
- Verified tensor shape transformations
- Confirms feature compression to match qubit count

### 2. Quantum Neural Network

`qnn_test.py`

- Standalone QNN implementation
- Validates:
  - input shape → `[batch, 2]`
  - output shape → `[batch, 1]`

- Uses Qiskit’s `EstimatorQNN`

### 3. Hybrid Model

`hybrid_test.py`

- Full CQ-CNN forward pipeline
- Combines CNN + QNN
- Validates end-to-end execution and tensor flow

---

## Key Insights

- Classical CNN reduces high-dimensional image data to a **low-dimensional quantum-compatible representation**
- The number of output features from the classical model must match the **number of qubits**
- The QNN acts as a **trainable transformation layer** replacing a classical dense layer
- The hybrid approach introduces:
  - potential expressivity advantages
  - but also training instability challenges

---

## Current Status

- [x] CNN architecture understood and reimplemented
- [x] QNN block isolated and tested
- [x] Hybrid forward pass verified
- [ ] Training loop analysis
- [ ] Small-scale training experiments
- [ ] Reproduction of paper results
- [ ] Model extensions for research contribution

---

## Setup

Create a virtual environment and install dependencies:

```bash
python -m venv qc_env
source qc_env/bin/activate

pip install torch torchvision
pip install qiskit qiskit-machine-learning
pip install numpy matplotlib scikit-learn
```

---

## Running Tests

### CNN test

```bash
python my_model_test.py
```

### QNN test

```bash
python qnn_test.py
```

### Hybrid model test

```bash
python hybrid_test.py
```

---

## Future Work

- Implement and analyze the training loop
- Perform experiments on reduced datasets
- Compare 2-qubit vs 3-qubit performance
- Investigate convergence issues in QNN training
- Extend architecture for improved stability and accuracy
- Explore alternative quantum feature maps and ansätze

---

## Research Direction

This project is part of a broader goal to explore:

> Hybrid quantum-classical models for medical imaging and their stability across datasets.

---

## Author

Mustakim Ahmed Hasan
Computer Science and Engineering
North South University

---

## Notes

This repository is intended for learning, experimentation, and research preparation.
It is not an official reproduction of the original authors’ implementation.
