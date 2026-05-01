# Learning Reduced-Round Cipher Behavior using Machine Learning
## Advanced Cryptography (AC) Project | Final Submission

**Team Members:**
- **Saloni** (MT25081)
- **Amisha Patel** (MT25061)

---

## 📑 Table of Contents
1. [Project Overview](#1-project-overview)
2. [Executive Summary](#2-executive-summary)
3. [Target Ciphers & Specifications](#3-target-ciphers--specifications)
4. [Methodology & ML Architectures](#4-methodology--ml-architectures)
5. [Key Findings & Performance Analysis](#5-key-findings--performance-analysis)
6. [Conclusion](#6-conclusion)
7. [Installation & Usage](#7-installation--usage)

---

## 1. Project Overview
This research explores the intersection of **Deep Learning** and **Symmetric Cryptography**. By treating modern block ciphers as complex non-linear mappings, we evaluate the effectiveness of various Machine Learning (ML) architectures in approximating ciphertext bits from plaintexts across reduced-round variants. Our goal is to empirically measure the "diffusion" rate and identify structural vulnerabilities in these ciphers using data-driven methods.

## 2. Executive Summary
- **Primary Finding:** Round 1 and Round 2 outputs of almost all modern ciphers exhibit significant non-random bias detectable by ML models.
- **Model Efficiency:** **Multi-Layer Perceptrons (MLP)** and **Logistic Regression** were the most effective at identifying linear and non-linear biases in SPN-based ciphers.
- **Vulnerability Peak:** **PRESENT** and **XOODOO** showed the highest vulnerability, with bitwise accuracies exceeding **95%** at Round 1.
- **Security Threshold:** All 8 tested ciphers achieved near-perfect diffusion (50% accuracy/random state) by **Round 5**, confirming the robustness of their full-round designs.

## 3. Target Ciphers & Dataset Specifications
We selected 8 diverse ciphers representing the core pillars of modern symmetric design. The training/testing split is consistently **80/20**, with dataset sizes optimized for each cipher's complexity.

| Cipher        | Design Philosophy                     | Block Size | Total Samples | Train/Test Split |
| :------------ | :------------------------------------ | :--------- | :------------ | :--------------- |
| **AES**       | Substitution-Permutation Network (SPN)| 128-bit    | 20,000        | 16k / 4k         |
| **SIMON**     | Feistel Network (Lightweight)         | 32-bit     | 200,000       | 160k / 40k       |
| **SPECK**     | ARX (Addition-Rotation-XOR)           | 32-bit     | 200,000       | 160k / 40k       |
| **PRESENT**   | Lightweight SPN                       | 64-bit     | 80,000        | 64k / 16k        |
| **KATAN**     | LFSR-based Hardware Cipher            | 32-bit     | 150,000       | 120k / 30k       |
| **XOODOO**    | Permutation-based (Xoodoo-p)          | 64-bit     | 70,000        | 56k / 14k        |
| **TRIVIUM**   | Hardware-oriented Stream Cipher       | 64-bit     | 60,000        | 48k / 12k        |
| **TinyJAMBU** | Lightweight Authenticated Cipher      | 64-bit     | 60,000        | 48k / 12k        |

> **Note:** While the split is 80/20, certain deep learning models (MLP, CNN) may employ internal capacity caps during training to ensure computational efficiency on standard hardware.

## 4. Methodology & ML Architectures

### A. Data Pipeline
- **Dataset Size:** 150,000+ samples per cipher/round configuration.
- **Feature Augmentation:** Plaintext bits are expanded using bitwise interaction terms (XORs) to assist models in capturing the internal logic of the cipher.
- **Normalization:** `StandardScaler` is employed to ensure stable gradient descent for neural networks.

### B. Machine Learning Models
1. **MLP (Multi-Layer Perceptron):** A dense 128-64 neuron architecture using **ReLU** activation. Optimized for capturing the non-linear "Confusion" property provided by S-Boxes.
2. **CNN (1D Convolutional Neural Network):** Implemented in **PyTorch**. Uses sliding kernels to detect local bit-level dependencies and spatial hierarchies.
3. **Random Forest:** An ensemble of 100+ decision trees. Highly effective at identifying patterns in **ARX**-based ciphers like SPECK.
4. **Logistic Regression:** Serves as the baseline for measuring **Linear Cryptanalysis** susceptibility.

## 5. Key Findings & Performance Analysis

### Champion Models by Cipher (Round 1)
| Cipher      | Champion Model  | Peak Accuracy | Observation                                           |
| :---------- | :-------------- | :------------ | :---------------------------------------------------- |
| **PRESENT** | MLP             | **96.40%**    | Critical vulnerability in early S-box layers.         |
| **XOODOO**  | Logistic        | **95.46%**    | High linear bias at Round 1.                          |
| **SIMON**   | MLP             | **93.65%**    | Vulnerable Feistel structure at R1/R2.                |
| **SPECK**   | Random Forest   | **79.91%**    | ARX structure resists linear models but yields to RF. |
| **AES**     | MLP             | **70.30%**    | Most robust at R1 due to heavy Diffusion layer.       |

### Learnability Trends
- **ARX vs. SPN:** ARX-based ciphers (SPECK, TinyJAMBU) show more "non-linear resistance" to simple neural networks compared to SPN ciphers at Round 1.
- **Diffusion Speed:** Ciphers like **PRESENT** drop from 96% to 61% accuracy between Round 1 and 2, showcasing extremely fast diffusion.
- **CNN Paradox:** 1D-CNNs performed moderately well but were often outperformed by MLPs, suggesting that cryptographic bit-diffusion is a global rather than a local phenomenon.

## 6. Conclusion
This project demonstrates that while modern cryptographic standards are secure in their full-round implementations, their **reduced-round variants** are highly susceptible to Machine Learning based approximation. The ability of an **MLP** to achieve **96% bitwise accuracy** on Round 1 of a standard cipher highlights the critical role of iterative "Confusion and Diffusion" rounds in preventing statistical leakages.

## 7. Installation & Usage

### 🚀 Setup
```powershell
# Create and activate environment
python -m venv .venv
.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

### 🛠️ Execution
**Run Full Comparative Suite:**
```powershell
python run_all.py
```

**Run Single Cipher Experiment (e.g., AES):**
```powershell
python experiments/main_aes.py
```


