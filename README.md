# INDRA-Sat-Diff: A Deep Learning Framework for Climate Forecasting

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**INDRA-Sat-Diff** is a comprehensive, PyTorch Lightning-based framework for training and evaluating deep learning models for high-resolution climate and weather forecasting. It implements a sophisticated three-stage training pipeline centered around a latent diffusion model, guided by a physics-informed alignment model to produce physically plausible and accurate predictions.

The architectural design and training methodology of this framework are heavily inspired by the principles and implementation of **PreDiff** ("Precipitation Nowcasting with Latent Diffusion Models," Gao et al., NeurIPS 2023). Our goal is to provide a refactored, accessible, and user-friendly implementation of these powerful techniques, complete with a streamlined workflow for custom datasets.

This framework is designed to be highly configurable and extensible, allowing researchers and practitioners to easily adapt it to different datasets and forecasting challenges.

## ✨ Core Features

*   **Modular Three-Stage Training Pipeline:** Ensures robust and stable training by decoupling the learning of data representation, physical constraints, and the forecasting process itself.
*   **Powerful Command-Line Interface:** Simple and intuitive commands (`preprocess`, `train`, `forecast`) for managing the end-to-end workflow.
*   **Highly Configurable:** A clean and powerful configuration system allows users to define their entire experiment—from data paths and model architecture to training parameters—in a single YAML file.
*   **Built-in Visualization:** Automatically generates static plots and animated GIFs of forecast outputs for easy analysis and presentation.
*   **Extensible Architecture:** Designed from the ground up to allow for the integration of new model architectures, datasets, and physical constraints.

## 🏛️ Architecture Overview

The framework's core is a **Latent Diffusion Model** that learns to reverse a noise-adding process in a compressed latent space. The training is performed in three distinct stages to ensure stability and performance:

1.  **Stage 1: Variational Autoencoder (VAE) Training**
    *   A VAE is trained to compress high-resolution climate data into a lower-dimensional, information-rich latent space. This allows the subsequent diffusion model to operate much more efficiently.

2.  **Stage 2: Knowledge Alignment Model Training**
    *   A separate model is trained to predict a specific physical property (e.g., the average precipitation intensity over the forecast period) directly from a *noisy* latent state and a corresponding timestep. This model learns a representation of the physical constraint.

3.  **Stage 3: Latent Diffusion Model Training**
    *   The main forecasting model, a U-Net based on the Cuboid Transformer, is trained to denoise a random latent state, conditioned on the previous weather states. During sampling (forecasting), the gradient from the pre-trained **Alignment Model** is used to guide the diffusion process, steering the forecast towards physically consistent and realistic outcomes.

## 🚀 Getting Started

This guide will walk you through setting up your environment, installing the framework, and running the provided example.

### Prerequisites

*   Git
*   Python 3.9+
*   An NVIDIA GPU with CUDA support is highly recommended for training.

### Installation Steps

All commands should be run from your terminal.

**1. Clone the Repository**
First, clone this repository to a location on your machine.

```bash
git clone https://github.com/vishwajitsarnobat/INDRA-Sat-Diff
cd INDRA-Sat-Diff
```

**2. Install `uv`**
We use `uv` for fast and reliable environment and package management. Install it using the official script:

```bash
# On macOS, Linux, and Windows Subsystem for Linux (WSL)
curl -LsSf https://astral.sh/uv/install.sh | sh
```
Follow the on-screen instructions to add `uv` to your shell's PATH.

**3. Create and Activate a Virtual Environment**
It is standard practice to work within a virtual environment.

```bash
# This creates a .venv folder in the repository's root directory
uv venv

# Activate the environment (command depends on your shell)
source .venv/bin/activate
```

**4. Install the Framework and Dependencies**
Install the `climate-forecast` package in "editable" mode (`-e`). This is the recommended approach, as it allows you to modify the source code and have the changes immediately take effect without reinstalling.

```bash
# This command reads the pyproject.toml and installs everything
uv pip install -e .
```
You are now ready to use the framework!

## ⚡ Quick Start: Running the Example

The best way to get started is to run the self-contained project in the `examples/` directory. This directory is a template for how you should structure your own independent projects.

**1. Navigate to the Example Directory**
```bash
cd examples
```

**2. Follow the Tutorial**
Inside the `examples/` directory, you will find a detailed `README.md` file. This is your primary guide. It will walk you through:
*   Setting up the necessary data folders.
*   Configuring the `config.yaml` for your dataset.
*   Running the `preprocess`, `train`, and `forecast` commands.

**Follow the instructions in `examples/README.md` to run your first complete experiment.**

## ⚙️ Configuration Philosophy

The framework operates on a simple and powerful configuration principle:

1.  **Framework Defaults (`climate_forecast/configs/train.yaml`):** This file, located inside the installed package, contains a comprehensive set of default parameters for every component of the framework. You should **not** edit this file directly.

2.  **User Configuration (`your_project/config.yaml`):** When you run a command, you provide your own YAML file. This file only needs to contain the parameters you wish to **override**. The framework will automatically load the defaults and merge your specified values on top.

The `examples/config.yaml` file is a perfect template for a minimal, clean user configuration.

## 📁 Repository Structure

```
INDRA-Sat-Diff/
├── climate_forecast/   # The core installable Python package source code.
│   ├── configs/        # Default configuration templates.
│   ├── datasets/       # Data loading, processing, and visualization logic.
│   ├── diffusion/      # Diffusion model and alignment logic.
│   ├── models/         # Core model architectures (U-Net, Transformers).
│   ├── pipelines/      # Logic for the train, preprocess, forecast commands.
│   └── training/       # PyTorch Lightning modules for each training stage.
│
├── examples/           # A self-contained user project template and tutorial.
│   ├── main.py         # A user-facing CLI script to run the pipelines.
│   └── config.yaml     # A minimal user config file for the example.
│
├── pyproject.toml      # Package definition and dependencies.
└── README.md           # You are here.
```

## 📄 Acknowledgements and Credits

This work is a re-implementation and refactoring of the architecture and methods introduced in the **PreDiff** project. We extend our sincere gratitude to the original authors for their foundational research and for making their work public.

*   **PreDiff: Precipitation Nowcasting with Latent Diffusion Models**
    *   Zhihan Gao, Xingjian Shi, Boran Han, Hao Wang, Xiaoyong Jin, Danielle Maddix Robinson, Yi Zhu, Mu Li, Yuyang Bernie Wang.
    *   **Paper:** [NeurIPS 2023](https://openreview.net/forum?id=Vp0045m12d)
    *   **Original Repository:** [https://github.com/gaozhihan/PreDiff](https://github.com/gaozhihan/PreDiff)

Several components within our framework are adapted from other outstanding open-source projects. We are grateful for their contributions to the community. These include:
*   **Stable Diffusion:** Implementations of the diffusion process and VAE are adapted from the work by CompVis at LMU Munich. ([GitHub](https://github.com/CompVis/stable-diffusion))
*   **Hugging Face Diffusers:** The underlying VAE building blocks (`unet_2d_blocks.py`, `resnet.py`, etc.) are adapted from the `diffusers` library. ([GitHub](https://github.com/huggingface/diffusers))
*   **Earthformer:** The core `CuboidTransformer` architecture is adapted from the work by Gao et al. ([GitHub](https://github.com/amazon-science/earth-forecasting-transformer))
*   **OpenAI:** Portions of the diffusion utilities and model components are adapted from OpenAI's guided-diffusion and improved-diffusion repositories.

## 📜 License

This project is licensed under the MIT License. Please see the `LICENSE` file for details. Note that the licenses of the third-party libraries we have adapted code from (such as Apache 2.0) may also apply to specific modules. We have retained original copyright notices where applicable.