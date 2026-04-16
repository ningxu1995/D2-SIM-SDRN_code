# SDRN: Sparsity-Driven Reconstruction Network for D²-SIM

[![Python 3.13](https://img.shields.io/badge/python-3.13-blue.svg)](https://www.python.org/downloads/release/python-3130/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

This repository contains the official PyTorch implementation of the **Sparsity-Driven Reconstruction Network (SDRN)**, the core computational engine for **Dual-frame Diffractive Structured Illumination Microscopy (D²-SIM)**. 

This code supports the paper:
> **"Standalone  dual-frame diffractive structured illumination enables sustained live-cell nanoscopy"** > 2026 （under review）.

## 📖 Overview
Conventional live-cell SIM typically requires 9 to 25 phase-shifted frames, severely limiting temporal resolution and inducing phototoxicity. D²-SIM mathematically compresses this sampling requirement to merely **two sparse inputs** (one widefield and one diffractive lattice frame) to achieve a 29.0-ms instantaneous reconstruction. 

To strictly preclude unphysical structural "hallucinations" common in purely data-driven models, the SDRN is trained on a rigorously noise-calibrated physical forward model utilizing a composite loss function ($L_1$, SSIM, and Fourier Domain Loss). **No Generative Adversarial Networks (GANs) or Perceptual (VGG) losses are used**, ensuring absolute physical fidelity for quantitative biophysics.

## ⚙️ System Requirements
* **OS:** Linux (Ubuntu 20.04/22.04 recommended) or Windows 11
* **GPU:** NVIDIA GPU with at least 12 GB VRAM (Tested on RTX 3090 24GB)
* **Python:** v3.13
* **Framework:** PyTorch (with CUDA support)

## 🛠️ Installation & Environment Setup
Clone the repository and install the required dependencies:

```bash
git clone [https://github.com/](https://github.com/)[ningxu1995]/D2-SIM-SDRN.git
cd D2-SIM-SDRN

# It is highly recommended to use a Conda virtual environment
conda create -n sdrn_env python=3.13
conda activate sdrn_env

# Install dependencies
pip install -r requirements.txt

(Note: Please ensure your PyTorch version matches your local CUDA toolkit version.)

📂 Repository Structure
Plaintext

D2-SIM-SDRN/
├── data/                       # Toy dataset for quick validation
│   ├── test/                   # Synthetic noisy inputs and Ground Truths
│   └── experiment/             # Raw live-cell acquisition frames
├── models/                     # Network architectures and constraints
│   ├── Models.py               # U-Net backbone definition
│   ├── Models_DRLN.py          # Residual Channel Attention Network (RCAN)
│   └── MS_SSIM_L1.py           # Physics-constrained composite loss functions
├── pre_trained_weights/        # Optimal model weights ready for inference
│   └── SDRN_best_model.pth     
├── results/                    # Output directory for reconstructed images
├── utils/                      # Helper scripts (I/O, physical forward-modeling)
├── options.py                  # Global hyperparameters configuration
├── Train.py                    # Main script for SDRN training
├── Test.py                     # Script for synthetic data validation
└── Run.py                      # Script for one-click experimental data inference
🚀 Quick Start: Inference on Toy Dataset
We provide a minimal toy dataset and pre-trained weights for immediate validation.

1. Validate on synthetic testing data:

Bash

python Test.py
Outputs will be saved in results/test/. You can compare the network predictions against the provided Ground Truths.

2. Reconstruct real experimental live-cell frames:

Bash

python Run.py
Outputs will be saved in results/experiment/, generating super-resolved nanoscale topologies from the two sparse empirical frames.

🧠 Model Training (Optional)
If you wish to train the SDRN from scratch using your own physical forward-projected data:

Configure your dataset paths and hyperparameters in options.py.

Run the training engine:

Bash

python Train.py
📊 Data Availability
The complete extensive training datasets (physically forward-projected from BioSR), alongside full raw experimental acquisition sequences, are persistently archived on Zenodo:
🔗 [Insert Zenodo DOI Link Here]

📄 License
This project is licensed under the MIT License - see the LICENSE file for details.