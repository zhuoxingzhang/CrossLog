# CrossLog: Cross-system Anomaly Detection from System Logs

[![License: CC BY-NC-ND 4.0](https://img.shields.io/badge/License-CC%20BY--NC--ND%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-nd/4.0/)

---

## Overview

**CrossLog** is a graph-based framework for **cross-system log anomaly detection at the code-file level**. It transfers anomaly knowledge learned from multiple source systems to a newly deployed target system, enabling effective detection even when labeled data for the target system is scarce.

![CrossLog Framework](CrossLog%20Code/crosslog-framwork.png)

CrossLog addresses two core challenges in cross-system log anomaly detection:

1. **Semantic heterogeneity** — different systems describe similar behaviors with different vocabularies, templates, and levels of granularity.
2. **Structural ambiguity** — structurally similar local patterns do not necessarily share the same semantics across systems, while semantically similar behaviors may appear through different structural forms.

To tackle these, CrossLog introduces:

- **Event Abstraction Algorithm (EAA)** — organizes log events into a 3-level semantic hierarchy and iteratively adjusts abstraction granularity through a coupling-control mechanism, constructing an abstract event pool that balances cross-system commonality and system diversity.
- **Abstraction-guided Substructure Mining (AGSM)** — learns a bank of transferable structural patterns in the abstract event space, guided by abstraction-anchored semantic signals to suppress system-specific bias.

Following a **pre-training and fine-tuning** paradigm, CrossLog learns complementary system-agnostic and system-specific representations from source systems, then efficiently adapts to the target system with only a small amount of labeled data.

---

## Key Results

CrossLog is evaluated on three real-world log datasets: **Novel**, **Forum**, and **Halo**.

| Dataset | Fine-tuning Size | F1 | Recall | Precision | PR-AUC |
|---------|:---:|:---:|:---:|:---:|:---:|
| Novel   | 0.1k | 0.80 | 0.77 | 0.83 | 0.84 |
| Novel   | 1k   | 0.92 | 0.92 | 0.93 | 0.92 |
| Forum   | 0.1k | 0.75 | 0.72 | 0.78 | 0.80 |
| Forum   | 1k   | 0.87 | 0.79 | 0.97 | 0.89 |
| Halo    | 0.1k | 0.81 | 0.80 | 0.83 | 0.87 |
| Halo    | 1k   | 0.94 | 0.93 | 0.96 | 0.97 |

CrossLog consistently outperforms all baselines, improving F1 by **8–21 percentage points** and PR-AUC by **5–13 percentage points** over the second-best method.

---

## Repository Structure

```
CrossLog/
├── CrossLog Code/
│   ├── crosslog-framwork.png          # Framework diagram
│   ├── args_config.py                 # Hyperparameter configuration
│   ├── event_abs_alg.py               # Event Abstraction Algorithm (EAA)
│   ├── dataloader.py                  # Graph loading and dataset construction
│   ├── GraphDataloader.py             # Lazy-loading dataset for large-scale data
│   ├── model.py                       # CrossLog model (GNN, AGSM, Pipeline)
│   ├── train.py                       # Source-system pre-training
│   ├── finetune_and_anomaly_infer.py  # Target-system fine-tuning and inference
│   ├── split_dataset.py               # Train/val/test split utilities
│   ├── utils.py                       # Metrics, threshold search, model I/O
│   └── main.py                        # Full pipeline entry point
├── data/                              # Pre-processed embeddings and event abstraction maps
│   ├── final_towards_target_{dataset}_event_abstraction_c{coupling}.json
│   ├── unified_level_event_abstraction_embedding_c{coupling}.json
│   ├── unified_exception_embedding.json
│   └── unified_filename_embedding.json
├── {dataset}_dataset/                 # Raw graph files for forum / halo / novel
├── {dataset}_data/                    # Split index files per dataset ID
├── models/                            # Saved model checkpoints
└── log/                               # Inference result logs
```

---

## Requirements

```
python >= 3.9
torch >= 2.0
torch_geometric
networkx
scikit-learn
numpy
tqdm
```

Install PyTorch Geometric following the [official instructions](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) for your CUDA version, then:

```bash
pip install networkx scikit-learn tqdm
```

---

## Data Preparation

### Datasets

CrossLog uses three open-source system log datasets: **Forum**, **Halo**, and **Novel**. Each raw log file is pre-processed into a graph file containing trace-level invocation information. Place the graph files under `{dataset}_dataset/`.

### Preprocessing

Generate train/val/test split index files:

```bash
cd "CrossLog Code"
python split_dataset.py
```

This creates split files under `{dataset}_data/{dataset_id}/` for both source and target roles.

### Event Abstraction

Run the Event Abstraction Algorithm to construct the abstract event pool. Output JSON files should be placed under `data/`:

```bash
python event_abs_alg.py
```

---

## Usage

### Full Pipeline

The complete experimental pipeline — covering all source/target combinations and fine-tuning scales — can be launched with:

```bash
cd "CrossLog Code"
python main.py
```

This iterates over all three leave-one-out cross-system scenarios and five fine-tuning sizes (0.5k, 1k, 2k, 3k, 4k graphs).

### Step 1: Pre-training on Source Systems

```bash
python train.py \
  --source_dataset novel,halo \
  --target_dataset forum \
  --dataset_id 51
```

The best checkpoint is saved to `./models/towards_target_{target}_{dataset_id}_pretrain_checkpoint.pth`.

### Step 2: Fine-tuning and Inference on Target System

```bash
python finetune_and_anomaly_infer.py \
  --source_dataset novel,halo \
  --target_dataset forum \
  --dataset_id 51 \
  --num_finetuning 500 \
  --ratio_finetuning 0.2
```

Results (F1, Recall, Precision, PR-AUC, ROC-AUC, and inference time) are written to `./log/test_results_{dataset_id}.txt`.

### Zero-shot Evaluation

Set `--num_finetuning 0` to skip fine-tuning and evaluate the pre-trained model directly on the target system.

---

## Method Details

### Event Abstraction Algorithm

Each log event is abstracted by an LLM into three semantic levels:

- **Level 1** — concrete operation type (e.g., *Query*)
- **Level 2** — business semantic type (e.g., *Information Retrieval*)
- **Level 3** — system intent (e.g., *Deliver Content*)

Starting from Level 3, the algorithm iteratively refines events to lower abstraction levels until the coupling between source systems falls below a threshold τ. The resulting abstract event pool balances cross-system semantic commonality and system-specific diversity.

### Model Architecture

CrossLog uses a **two-branch** design:

- **System-agnostic branch** — a GNN encoder + Gated Global Attention module produces substructure representations in the abstract event space; AGSM then selects appropriate transferable structure patterns under semantic guidance, producing `Z_agn`.
- **System-specific branch** — a separate GNN processes the original log graphs to capture system-dependent characteristics, producing `Z_spe`.

During fine-tuning, the system-agnostic branch and system classifier are **frozen**; only the system-specific branch and anomaly detector are updated.

### Training Objective

$$\mathcal{L} = \lambda_1 \mathcal{L}_{sys} + \lambda_2 \mathcal{L}_{ano} + (1 - \lambda_1 - \lambda_2)\, I(Z_{spe}; Z_{agn})$$

where $\mathcal{L}_{sys}$ is the system classification loss, $\mathcal{L}_{ano}$ is the anomaly detection loss, and $I(Z_{spe}; Z_{agn})$ is a mutual information term that encourages disentanglement between the two representations.

---

## License

This work is licensed under the [Creative Commons BY-NC-ND 4.0 International License](https://creativecommons.org/licenses/by-nc-nd/4.0/).
