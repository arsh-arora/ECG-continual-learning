# ECG Continual Learning Framework

This repository contains a comprehensive framework for continual learning on ECG data using the PTB-XL dataset. The framework implements various continual learning approaches including Task-Incremental Learning (TiL), Class-Incremental Learning (CiL), Learning without Forgetting (LwF), Elastic Weight Consolidation (EWC), and Synaptic Intelligence (SI).

## File Overview

### 1. CL_PTBXL.py
The main implementation file for continual learning on the PTB-XL dataset. This file handles:
- Data loading and preprocessing of the PTB-XL dataset
- Creation of super-class and sub-class diagnostic labels
- Implementation of CNN, ResNet, and ViT model architectures
- Training and evaluation functions
- Implementation of LwF, EWC, and SI continual learning methods
- Generation of classification reports and performance metrics

Key features:
- Multi-label classification for ECG diagnostic classes
- Data normalization per channel
- Class weighting to handle imbalanced datasets
- Custom loss functions for continual learning methods
- Comprehensive evaluation metrics including Macro F1-score

### 2. ablation_study.py
Performs ablation studies on CNN, ResNet, and ViT model families for the super-diagnostic task on PTB-XL. This script:
- Imports data splits, class weights, and utilities from CL_PTBXL.py
- Trains variants of each model family:
  - CNN: x2, x4, x8, x16 (width multipliers)
  - ResNet: 18, 34, 50, 101 (basic-block variants)
  - ViT: Small, Base, Large, Huge (1D ViT with patchifying on time axis)
- Logs model parameters, training time, and Macro-F1 scores to 'ablation_results.csv'

Key features:
- Reproducibility settings with configurable seed
- Mixed precision training support
- Multi-GPU training support with strategy scope
- Automatic memory cleanup between training runs

### 3. avalance_cl.py
Avalanche backend implementation for continual learning with ECG data. This file provides:
- PyTorch implementations of 1D CNN, ResNet, and ViT backbones
- Data adapters for converting NumPy arrays to PyTorch datasets
- Multi-head model architecture for Task-Incremental Learning
- Benchmark creation for both Task-Incremental Learning (TiL) and Class-Incremental Learning (CiL)
- Integration with Avalanche continual learning strategies (Naive, EWC, Synaptic Intelligence)
- Evaluation metrics for continual learning scenarios

Key features:
- Support for both single-label and multi-label classification
- Flexible model specification with parametric model sizes (Small, Medium, Large, Huge)
- Task-aware model switching for multi-head architectures
- Comprehensive evaluation with accuracy and Macro F1-score metrics

### 4. avalance_ablation.py
Avalanche runner for ECG continual learning that performs ablation studies using the Avalanche framework. This script:
- Runs both Task-Incremental Learning (TiL) and Class-Incremental Learning (CiL) scenarios
- Tests multiple model families (CNN, ResNet, ViT) with different sizes
- Evaluates continual learning methods (Naive, EWC, Synaptic Intelligence)
- Outputs results to 'results_ecg_cl_avalanche.csv'

Key features:
- Systematic evaluation of continual learning methods
- Forgetting metrics calculation
- Support for both TiL and CiL scenarios
- CSV output for easy result analysis

## Usage

### Data Preparation
Ensure the PTB-XL dataset is available in the expected directory structure. The framework expects:
```
ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3/
├── ptbxl_database.csv
├── scp_statements.csv
└── (ECG record files)
```

### Running the Framework
1. Execute `CL_PTBXL.py` to run the base continual learning experiments
2. Run `ablation_study.py` to perform ablation studies on model architectures
3. Execute `avalance_ablation.py` to run continual learning experiments with the Avalanche framework

## Model Architectures

### CNN
1D Convolutional Neural Networks with configurable width and depth.

### ResNet
1D Residual Networks with basic blocks and configurable depth (18, 34, 50, 101).

### ViT (Vision Transformer)
1D Vision Transformer adapted for ECG signals with patch embedding and transformer encoder blocks.

## Continual Learning Methods

### LwF (Learning without Forgetting)
Preserves knowledge from previous tasks by using knowledge distillation.

### EWC (Elastic Weight Consolidation)
Protects important weights for previous tasks using Fisher information.

### SI (Synaptic Intelligence)
Measures weight importance during training to prevent catastrophic forgetting.

### Avalanche Strategies
- Naive (no continual learning)
- EWC (Elastic Weight Consolidation)
- Synaptic Intelligence

## Output Files
- `ablation_results.csv`: Results from the ablation study
- `results_ecg_cl_avalanche.csv`: Results from Avalanche-based continual learning experiments

## Requirements
- Python 3.7+
- TensorFlow 2.x
- PyTorch 1.7+
- Avalanche
- scikit-learn
- pandas
- numpy
- wfdb (for PTB-XL dataset)
# ECG-continual-learning
