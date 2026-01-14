Here is the documentation formatted in clean, professional Markdown, ready to be copied directly into a GitHub `README.md` or a Wiki page.

---

# Task 1: Centralized Baseline Training

| Metadata | Details |
| --- | --- |
| **Owner** | Sauryan |
| **Milestone** | 3 |
| **Status** | ✅ Complete |

## 1. Overview

This document details the process for running **centralized fine-tuning** using the official NicheFormer model implementation with pretrained checkpoints. It covers the setup environment, how to handle pretrained weights, execution commands, and specific improvements made to the shared integration code (model wrapper).

---

## 2. Setup & Installation

### Environment Setup

To ensure compatibility, a separate environment was created for the "real" NicheFormer integration.

````md
1. **Create and activate environment:**
```bash
conda create -n nicheformer_real python=3.9 -y
conda activate nicheformer_real
````

2. **Clone the NicheFormer repository:**
   From the root of this project, clone the official repo:

```bash

git clone https://github.com/theislab/nicheformer.git
cd nicheformer
```

3. **Install NicheFormer + dependencies:**
   Install repo dependencies (if present) and then install NicheFormer into the environment:

```bash
# If requirements.txt exists
pip install -r requirements.txt

# Install the package (editable)
pip install -e .
```

> If the repo provides `environment.yml` instead of `requirements.txt`, use:

```bash
conda env update -n nicheformer_real -f environment.yml
```

4. **Verify NicheFormer installation:**
   Run the following command to confirm the package is importable:

```bash
python -c "import nicheformer; print('nicheformer import OK'); print('version:', getattr(nicheformer,'__version__','unknown'))"
```

```
::contentReference[oaicite:0]{index=0}
```




### Pretrained Checkpoint Setup

**Important:** The pretrained weights are required for transfer learning.

1. **Download Source:**
* **Official Mendeley Data:** [NicheFormer Pretrained Weights](https://data.mendeley.com/preview/87gm9hrgm8?a=d95a6dde-e054-4245-a7eb-0522d6ea7dff)


2. **File Placement:**
* Download the file.
* **Rename** the file to: `nicheformer_pretrained.ckpt`
* **Move** it to the following directory (create directory if missing):
```text
data/pretrained/nicheformer_pretrained.ckpt

```





---

## 3. Workflow: `scripts/run_centralized.py`

The centralized training script manages the end-to-end workflow for non-federated learning.

### Key Steps

1. **Data Loading:**
* Reads `train`, `val`, and `test` splits from all `client_*` folders in `data/processed/clients/`.
* Pools all client data into single DataFrames (`train_df`, `val_df`, `test_df`).
* Builds batches using the canonical gene order from `load_gene_list()`.


2. **Model Initialization:**
* If `--pretrained_path` is provided: Loads the official NicheFormer backbone.
* If not provided: Uses a placeholder backbone (for testing/dev only).


3. **Training:**
* Runs for `N` epochs.
* Evaluates on pooled validation data at the end of every epoch.


4. **Evaluation:**
* Calculates metrics on the **pooled test set**.
* Calculates per-client metrics (evaluates on each client's test split separately).


5. **Artifacts:**
* Saves the model, plots, and metrics to the specified output directory.



---

## 4. Usage

### CLI Arguments

| Argument | Description | Default |
| --- | --- | --- |
| `--data_dir` | Directory containing Milestone 2 processed data | `data/processed` |
| `--output_dir` | Directory to save results and artifacts | N/A |
| `--device` | Compute device (`cpu` or `cuda`) | `cuda` |
| `--epochs` | Number of training epochs | `5` |
| `--batch_size` | Training batch size | `128` |
| `--lr` | Learning rate | `1e-4` |
| `--fine_tune_mode` | Strategy: `head_only`, `partial`, or `full` | `head_only` |
| `--pretrained_path` | Path to the `.ckpt` file | `None` |
| `--include_spatial` | Flag to include x,y coordinates | `False` |

### Execution Commands

**Note:** The examples below use standard Linux line continuations (`\`). If using Windows CMD, replace `\` with `^`.

#### 1. Smoke Test (Sanity Check)

Run a quick 1-epoch test to validate the wiring and pipeline.

```bash
python scripts/run_centralized.py \
  --data_dir data/processed \
  --output_dir results/centralized_pretrained_smoke \
  --epochs 1 \
  --batch_size 128 \
  --lr 1e-4 \
  --fine_tune_mode head_only \
  --pretrained_path data/pretrained/nicheformer_pretrained.ckpt \
  --device cuda

```

#### 2. Main Run (Pretrained + Partial Fine-tuning)

The recommended configuration for actual training results.

```bash
python scripts/run_centralized.py \
  --data_dir data/processed \
  --output_dir results/centralized \
  --epochs 5 \
  --batch_size 128 \
  --lr 1e-4 \
  --fine_tune_mode full \
  --pretrained_path data/pretrained/nicheformer_pretrained.ckpt \
  --device cuda

```

---

## 5. Outputs

After a successful run, the `--output_dir` will contain:

* **`model_final.pt`**: The final fine-tuned PyTorch model weights.
* **`history.json`**: Epoch-by-epoch training and validation metrics.
* **`metrics.csv`**: Detailed pooled and per-client test results.
* **`eval_summary.json`**: High-level summary of test performance.
* **`config.yaml`**: A record of the configuration used for the run.
* **`plots/`**:
* `loss_curve.png`
* `accuracy_curve.png`
* `f1_macro_curve.png`
* `confusion_matrix.png`
* `per_client_accuracy.png`



---

## 6. Implementation Details & Fixes

Aftab’s responsibility: make the “real NicheFormer + weights” path possible (wrapper + checkpoint loading).

I updated the wrapper/integration so pretrained fine-tuning is real, stable, and consistent

### Fix 1: Backbone Output vs. Classifier Dimension Alignment

**Issue:** The pretrained NicheFormer encoder's internal embedding size often differed from the wrapper's default `hidden_dim` (e.g., 256), causing `RuntimeError: mat1 and mat2 shapes cannot be multiplied`.
**Solution:**

* Dynamically detected the actual output size of the backbone.
* Updated the classifier head initialization to match the backbone's output dimension automatically.

### Fix 2: Fine-Tuning Modes

**Issue:** Need to ensure that `head_only`, `partial`, and `full` modes correctly freeze/unfreeze parameters when a pretrained model is loaded.
**Solution:**

* **`head_only`**: Freezes the entire backbone; trains only the classifier head.
* **`partial`**: Unfreezes the last few layers of the backbone + the classifier head.
* **`full`**: Unfreezes all parameters.
* *Verification:* Confirmed by printing trainable parameter counts (count increases from `head_only` → `partial` → `full`).
