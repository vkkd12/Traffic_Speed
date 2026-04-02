# Traffic Speed Prediction: Core Models Overview

This document provides detailed insights into the three main deep learning models implemented in this repository, their architectural distinctions, and instructions on how to use them.

## 1. MT-STGIN: Multi-Task-Based Spatiotemporal Generative Inference Network
**Publication:** *Expert Systems with Applications* & *IEEE ITSC*

### Concept & Architecture
MT-STGIN addresses the heterogeneity of traffic patterns across different types of highway segments. Instead of treating the entire highway network as a homogenous graph, it frames traffic prediction as a **Multi-Task Learning** problem. It classifies prediction into three concurrent sub-tasks based on the road segment topologies:
1. **ETTG**: Entrance toll to gantry
2. **GTG**: Gantry to gantry
3. **GTET**: Gantry to exit toll

The model heavily integrates Graph Convolutional Networks (GCN) within an Encoder-Decoder structure framework. It employs dedicated `spatial_attention` and `temporal_attention` blocks to capture long-range spatial dependencies and dynamic temporal trends respectively.

### How It Is Used
- **Workspace Location:** `MT-STGIN/`
- **Execution Script:** `MT-STGIN/run_train.py`
- **Hyperparameters:** Configured in `MT-STGIN/models/hyparameter.py`. Adjust settings like `--epochs`, `--batch_size`, and network shape (e.g., `--num_blocks`, `--num_heads`).
- **Running:**
  Run `python run_train.py`. The console will prompt you to input `1` for the Training mode, or `0` for the Testing/Inference mode.

---

## 2. 3S-TBLN: Self-Supervised Spatio-Temporal Bilateral Learning Network
**Publication:** Submitted to *IEEE Transactions on Intelligent Transportation Systems (TITS)*

### Concept & Architecture
3S-TBLN operates on a philosophy of using **simple input variables** to achieve highly accurate predictions through a **Self-Supervised** embedding mechanisms. 
- **Bilateral Structure:** It employs a dual-branch structure (bilateral) isolating the spatial attention calculations and temporal attention calculations to prevent mutual interference before fusing them.
- **Self-Supervised Embeddings:** Instead of heavy manual feature engineering, it injects explicit Position Embeddings (SE) and Temporal Embeddings (TE) corresponding to the day-of-week, hour, and minute cycles to automatically trace regular traffic fluctuations.
- **Fusion Gate:** Features learned from the bilateral branches are eventually fused dynamically through customized fusion gates.

### How It Is Used
- **Workspace Location:** `3S-TBLN/`
- **Execution Script:** `3S-TBLN/run_train.py`
- **Hyperparameters:** Managed via `3S-TBLN/models/hyparameter.py`. Default setup expects inputs like 12 historical time steps targeting the next 6 or 12 future time steps predicting traffic speed variations.
- **Running:**
  Run `python run_train.py` inside the `3S-TBLN` folder. As with MT-STGIN, it will ask for input `1` to Train or `0` to Test models spanning Yinchuan, METR-LA, or PEMS-BAY datasets.

---

## 3. ST-ANet: Spatio-Temporal Attention Network for Dynamic Highway Network
**Publication:** *Journal of Computer Engineering*

### Concept & Architecture
ST-ANet is a specialized implementation designed specifically for dynamic, highly variable highway networks.
- **Generative Adversarial Design:** Based on the codebase (`ST-ANet/model/gan.py`), ST-ANet integrates a Generative Adversarial Network (GAN) foundation along with Graph Convolution routines. 
- **Dynamic Attention:** Uses spatio-temporal attention blocks (`ST-ANet/model/t_attention.py` and `lstm.py`) operating alongside the generator and discriminator modules to produce robust speed estimates that map realistically to observed traffic flows while penalizing unrealistic predictions.

### How It Is Used
- **Workspace Location:** `ST-ANet/`
- **Execution Script:** `ST-ANet/run.py`
- **Hyperparameters:** Configured in `ST-ANet/model/hyparameter.py` (e.g., configuring `gcn_output_size`, `hidden_size`, `input_length`, etc.). 

---

## General Usage & Environment Guidelines

To run any of the primary models smoothly, the developer environment needs to adhere to these constraints:

### 1. Environment Setup
To run these models, you should set up an isolated Python environment using Python's built-in `venv`:

```bash
# 1. Create a virtual environment using standard Python
python -m venv traffic_speed

# 2. Activate the virtual environment
# On Windows:
traffic_speed\Scripts\activate
# On Linux/Mac:
# source traffic_speed/bin/activate

# 3. Install required packages
pip install -r MT-STGIN/requirements.txt
```
*Note: Due to code implementations explicitly using TF 1.x features (such as `tf.variable_scope` and `tf.placeholder`), models are best suited for TensorFlow 1.13.1 / 1.14.1. If running on TensorFlow 2.x, the codes rely on `import tensorflow.compat.v1 as tf` and `tf.disable_v2_behavior()`. Ensure your base Python version is compatible with older TensorFlow versions (Python 3.7 is highly recommended).*

### 2. Dataset Preparation
Data is generally expected in `.csv` or `.h5` formats (inside folders like `data/YINCHUAN/`, `data/METR-LA/`). Ensure your dataset paths in the `hyparameter.py` scripts (`--file_train`, `--file_val`, `--file_test`) route to the correct data subsets.

### 3. Hyperparameter Tuning
Hyperparameters across the 3 models leverage Python's `argparse`. Instead of hardcoding values into the training scripts, you open their respective `hyparameter.py` files to uniformly change metrics such as `--learning_rate`, `--target_site_id`, `--batch_size`, or `--site_num`.

---

## Technical Stack & Dependencies

To accurately explain this project's foundation, here is the core technology stack and environments required:

### 1. Primary Deep Learning Framework
The core models (MT-STGIN, 3S-TBLN, ST-ANet) are built heavily on **TensorFlow**. 
- They specifically target **TensorFlow 1.13.1 / 1.14.1**. Features like `tf.variable_scope` and `tf.placeholder` are extensively used.
- They are backward-compatible with **TensorFlow 2.x** by using `import tensorflow.compat.v1 as tf` and `tf.disable_v2_behavior()`.

### 2. Baseline Model Frameworks
While the primary models are in TensorFlow, the repository contains over 15 baseline comparison models (stored in `3S-TBLN/baselines/` and `MT-STGIN/baseline/`). Many of these baselines rely on **PyTorch**. 
- Baselines like **AGCRN**, **DCRNN**, **Graph-WaveNet**, **STGCN**, and **MTGNN** utilize PyTorch (`torch`), reflecting their original implementations from other research papers.

### 3. Key Python Libraries
Across the project (as defined by various `requirements.txt` and project imports), the main libraries utilized include:
- **Numpy (`numpy`) & Pandas (`pandas`)**: Used extensively for dataset loading (`.csv`, `.h5`), feature processing, and array dimension manipulations.
- **Scipy (`scipy`)**: Heavily utilized in mathematical operations and graph processing, especially within the PyTorch-based baseline models (like adjacency matrix processing).
- **Matplotlib (`matplotlib`) & Seaborn (`seaborn`)**: Used extensively in `visualize.py` and `results/test.py` to plot predictive performance charts, line graphs comparing models, and heatmaps.
- **Argparse (`argparse`)**: Used across all `hyparameter.py` files to manage and dynamically parse execution flags (like `--epochs`, `--batch_size`, etc.).
- **Scikit-learn**: Used for traditional machine learning baselines (like SVR) and standardized metric computation.

### Hardware Consideration
- The project relies on CUDA architectures (`os.environ['CUDA_VISIBLE_DEVICES'] = '1'`) to manage training across 100+ road segments recursively. The original experiments utilized an NVIDIA Tesla V100S GPU with 32GB memory.