# DLOps Assignment-1

**Name:** <Your Name>  
**Roll Number:** <Your Roll Number>  
**Course:** DLOps Laboratory  

---

## Colab Notebook
🔗 <Colab Link>

## GitHub Repository
🔗 <GitHub Repository Link>

---

## Question 1(a): Deep Learning Models on MNIST and FashionMNIST

All models were trained using PyTorch with `pretrained=False`.  
Dataset split: **70% Train / 10% Validation / 20% Test**.  
AMP was enabled for GPU experiments.

---

### MNIST — Complete Results

| Batch | Optimizer | LR | Model | Test Acc (%) | Train Time (ms) | Final Val Acc (%) |
|------:|-----------|----|-------|--------------|-----------------|------------------|
| 16 | SGD  | 0.0010 | ResNet-18 | 99.11 | 395,215.03 | 99.07 |
| 16 | SGD  | 0.0010 | ResNet-50 | 98.81 | 714,489.22 | 98.93 |
| 16 | SGD  | 0.0001 | ResNet-18 | 97.09 | 389,802.45 | 97.14 |
| 16 | SGD  | 0.0001 | ResNet-50 | 96.11 | 713,614.67 | 96.33 |
| 16 | Adam | 0.0010 | ResNet-18 | 98.71 | 398,288.39 | 98.93 |
| 16 | Adam | 0.0010 | ResNet-50 | 98.41 | 747,441.42 | 98.59 |
| 16 | Adam | 0.0001 | ResNet-18 | 99.25 | 402,331.96 | 99.33 |
| 16 | Adam | 0.0001 | ResNet-50 | 98.88 | 749,854.28 | 98.87 |
| 32 | SGD  | 0.0010 | ResNet-18 | 98.41 | 336,615.53 | 98.66 |
| 32 | SGD  | 0.0010 | ResNet-50 | 98.49 | 644,463.93 | 98.66 |
| 32 | Adam | 0.0001 | ResNet-18 | 99.09 | 338,690.57 | 99.37 |
| 32 | Adam | 0.0001 | ResNet-50 | 97.23 | 662,719.75 | 97.57 |

---

### FashionMNIST — Complete Results

| Batch | Optimizer | LR | Model | Test Acc (%) | Train Time (ms) | Final Val Acc (%) |
|------:|-----------|----|-------|--------------|-----------------|------------------|
| 16 | SGD  | 0.0010 | ResNet-18 | 91.60 | 390,073.44 | 91.21 |
| 16 | SGD  | 0.0010 | ResNet-50 | 88.96 | 711,370.84 | 88.96 |
| 16 | SGD  | 0.0001 | ResNet-18 | 85.52 | 384,257.95 | 85.54 |
| 16 | SGD  | 0.0001 | ResNet-50 | 78.43 | 713,069.97 | 79.06 |
| 16 | Adam | 0.0010 | ResNet-18 | 91.66 | 400,073.03 | 91.86 |
| 16 | Adam | 0.0010 | ResNet-50 | 88.36 | 748,578.39 | 88.59 |
| 16 | Adam | 0.0001 | ResNet-18 | 92.07 | 413,977.75 | 91.79 |
| 16 | Adam | 0.0001 | ResNet-50 | 91.34 | 752,199.16 | 91.11 |
| 32 | SGD  | 0.0010 | ResNet-18 | 89.41 | 352,759.99 | 89.50 |
| 32 | SGD  | 0.0010 | ResNet-50 | 88.08 | 645,835.22 | 88.29 |
| 32 | Adam | 0.0001 | ResNet-18 | 91.57 | 340,907.69 | 91.34 |
| 32 | Adam | 0.0001 | ResNet-50 | 90.89 | 665,604.20 | 91.09 |

---

## Question 1(b): SVM Classification Results

| Dataset | Kernel | Test Accuracy (%) | Training Time (ms) |
|--------|--------|-------------------|--------------------|
| MNIST | Polynomial | 90.56 | 22,215.67 |
| MNIST | RBF | 93.80 | 11,161.82 |
| FashionMNIST | Polynomial | 82.67 | 11,526.57 |
| FashionMNIST | RBF | 85.30 | 8,846.74 |

---
## Question 2: CPU vs GPU Comparison (FashionMNIST)

**Fixed configuration:**  
Batch Size = 16, Learning Rate = 0.001

The objective of this experiment is to compare CPU and GPU execution in terms of
classification accuracy and computational complexity.

---

### CPU vs GPU Results

| Compute | Model | Optimizer | Test Accuracy (%) | FLOPs |
|--------|-------|-----------|-------------------|-------|
| CPU | ResNet-18 | SGD | 79.40 | 22.36 MFLOPs |
| CPU | ResNet-50 | SGD | Not completed | 47.06 MFLOPs |
| CPU | ResNet-18 | Adam | Not completed | Not completed |
| CPU | ResNet-50 | Adam | Not completed | Not completed |
| GPU | ResNet-18 | SGD | 56.20 | 22.36 MFLOPs |
| GPU | ResNet-50 | SGD | 67.40 | 47.06 MFLOPs |
| GPU | ResNet-18 | Adam | 33.85 | 22.36 MFLOPs |
| GPU | ResNet-50 | Adam | 81.15 | 47.06 MFLOPs |

---
