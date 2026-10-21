# Spatial Representations for Diffusion-Based Pathfinding

This repository contains the code for the CSM500 project investigating how explicit spatial representations affect a conditional diffusion planner on 64 × 64 mazes. The experiments compare raw occupancy information with directional, boundary-distance and topological landmark features, alongside a fixed random-channel control, using two U-Net capacities.

**Pretrained checkpoints are not included. Train a model before running its inference and evaluation cells.**

## Repository contents

| File or filename component | Purpose |
|---|---|
| `dataset generator.ipynb` | Generate A*-labelled training/validation data and a separate test set; inspect example maps and paths |
| `dataset/` | Dataset directory; training notebooks expect `pair_*.npz` directly inside it |
| `RAW_DIRECT` | Raw obstacle, start and goal channels |
| `RAW_NOISE4_DIRECT` | Raw channels plus four deterministic random channels |
| `DIRECTIONAL_DIRECT` | Raw channels plus four legal-movement channels |
| `BOUNDARY_DIRECT` | Raw channels plus four directional boundary-distance channels |
| `TOPOLOGICAL_LANDMARK4_DIRECT` | Raw channels plus four geodesic landmark proximity channels |
| `requirements.txt` | Python package installation requirements |

Experiment notebooks have the filename pattern `dataset_v2_10000_<REPRESENTATION>_DIRECT_x0_path_weighted_MATCHED_100ep_500t.ipynb`. The corresponding smaller model adds `_SMALL_UNET32` before `.ipynb`. The regular model uses `BASE_CHANNELS = 64`; the small model uses `BASE_CHANNELS = 32`.

## Environment and installation

The recorded RunPod environment uses Python **3.11.10**, PyTorch **2.4.1+cu124**, the PyTorch CUDA **12.4** build, and an **NVIDIA A100-SXM4-80GB**. This GPU is the recorded training device, not a minimum hardware requirement. A compatible NVIDIA GPU is recommended for full training and 500-step sampling.

Clone the repository:

```bash
git clone https://github.com/Emil0227/Project-CSM500.git
cd Project-CSM500
```

Create a virtual environment using Python 3.11.

Linux / RunPod:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the dependencies and register the notebook kernel:

```bash
python -m pip install -r requirements.txt
python -m pip check
python -m ipykernel install --user --name cognitive-map --display-name "Python (cognitive map)"
jupyter lab
```

Select **Python (cognitive map)** when opening a notebook. Launch JupyterLab from the repository root and keep the notebook working directory there so relative dataset paths resolve correctly.

The requirements file selects the CUDA 12.4 PyTorch build through the official PyTorch package index. It targets compatible NVIDIA Linux/Windows environments. CPU-only or macOS installations require an appropriate PyTorch build instead; see the [official installation instructions for previous versions](https://pytorch.org/get-started/previous-versions/). Dependency installation and numerical equivalence on other platforms have not been verified.

Check the active environment in a notebook:

```python
import os
import sys
import torch

print("Python:", sys.executable)
print("Working directory:", os.getcwd())
print("PyTorch:", torch.__version__)
print("PyTorch CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
```

## 1. Prepare the datasets

Open `dataset generator.ipynb`. It contains two generation sections:

| Section | Output directory | Samples | Seed |
|---|---|---:|---:|
| First generation cell | `dataset` | 10,000 | 42 |
| Later test generation cell | `test set` | 1,000 | 2026 |

The intervening cells visualise samples. Generate a dataset only when it is missing or when deliberately rebuilding it: rerunning generation writes to the same sample filenames. Use clean output directories when changing the sample count to avoid retaining older extra files.

If the repository's `dataset/` contains nested dataset versions, select the intended 10,000-sample version and set `DATASET_DIR` to the directory containing its `pair_*.npz` files. The loader does not search subdirectories. Generate `test set/` separately if it is absent.

The generator constructs 15 × 15 logical mazes, expands them to 64 × 64 grids, and introduces loops with `LOOP_PROB = 0.08`. Start and goal must be at least 30 Manhattan steps apart; accepted A* paths contain at least 80 cells and have a path-step/Manhattan-distance ratio of at least 1.5.

Each compressed `.npz` sample includes:

| Key | Meaning | Shape |
|---|---|---|
| `map` | Obstacle mask, one-hot start, one-hot goal | `(3, 64, 64)` |
| `path` | Binary A* reference path | `(1, 64, 64)` |
| `start`, `goal` | Row and column coordinates | `(2,)` each |
| `path_length` | Number of path cells, including endpoints | Scalar |

The training notebooks split the 10,000 samples into **9,000 training** and **1,000 validation** samples using `SPLIT_SEED = 42`. The separately generated 1,000-sample test set is used for final evaluation.

## 2. Choose a representation and train

Start with a `RAW_DIRECT` notebook, then run the other representations in separate kernels using the same dataset and experiment settings.

| Representation | Conditioning channels | Added information |
|---|---:|---|
| Raw | 3 | Obstacle, start and goal |
| Raw + Noise4 | 7 | Four fixed uniform random fields, seeded by sample index |
| Directional | 7 | Legal moves up, down, left and right |
| Boundary | 7 | Consecutive free-cell counts towards each obstacle/map boundary, normalised by 63 |
| Topological Landmark4 | 7 | Four obstacle-aware geodesic proximity fields, independent of the start, goal and target path |

Each representation is projected to 16 channels by a single learned **1 × 1 convolution**, then conditions a U-Net that predicts the clean path (`x0`). The reference path is transformed from `{0, 1}` to `{-1, 1}` during loading. Spatial feature channels are constructed by the loaders; separate labelled datasets are not required for each representation.

Default training settings:

| Setting | Value |
|---|---|
| Epochs | 100 |
| Batch size | 32 |
| Learning rate | `2e-4` |
| Optimizer | AdamW |
| Diffusion timesteps | 500 |
| Objective | Path-weighted mean squared error on `x0` |
| Path / background pixel weights | 5 / 1 |
| Gradient clipping | Maximum norm 1.0 |
| Checkpoint selection | Lowest validation loss |

Run cells in order: imports and configuration, dataset loading, model definitions, training setup, and the full training loop. Check `DATASET_DIR` and `CHECKPOINT_DIR` before starting.

Training writes `best.pt` and `latest.pt` to the experiment's local checkpoint directory. These files are generated locally and are not distributed with this repository. Rerunning training in the same directory can overwrite them. Loss curves and evaluation summaries appear in notebook output.

For a quick execution check, reduce epochs and evaluation sample count. Such a run is not a reproduction of the full experiment. Reducing batch size for a smaller GPU also changes the training configuration.

## 3. Evaluate the trained model

After training, run the best-checkpoint loading cell, reverse-diffusion definitions and evaluation cells. The final section, **Final Evaluation on Independent Test Set**, loads `best.pt`, reads `TEST_DATASET_DIR = "test set"`, and expects exactly 1,000 samples.

Sampling uses the reverse diffusion process. The resulting path mask is thresholded at **0.5**. Reported metrics are:

| Metric | Definition |
|---|---|
| Start included / Goal included | Fraction of predictions containing the respective endpoint |
| Collision-free | Fraction with no predicted path pixels on obstacles |
| Connected S → G | Fraction with a four-neighbour connection between start and goal in the predicted mask |
| Average collisions | Mean number of predicted pixels overlapping obstacles |
| Valid Path Success | Fraction satisfying endpoint inclusion, collision freedom and start-to-goal connectivity together |

These validity checks do not establish shortest-path optimality or require the entire predicted mask to be a single branch-free route.

### Landmark cache: separate training and test data

The landmark notebooks cache feature arrays in `LANDMARK_CACHE_DIR`. Cache filenames are based on sample filenames, so training and test data named `pair_00000.npz`, etc. must not share a cache directory.

**Before constructing `test_dataset` in the final evaluation cell**, add:

```python
LANDMARK_CACHE_DIR = "cache_topological_landmark_4ch_test"
os.makedirs(LANDMARK_CACHE_DIR, exist_ok=True)
prepare_landmark_cache(TEST_DATASET_DIR)
```

Place this after `TEST_DATASET_DIR = "test set"` and before `test_dataset = MazePathDataset(...)`. The supplied final evaluation cell does not perform this switch itself. Do not return to training/validation evaluation with the test cache selected; restore the training cache setting first. Rebuild the relevant cache whenever its source maps or landmark settings change.

## Reproducibility and troubleshooting

- Preserve the dataset, sorted filenames, split and randomness settings across comparisons. The notebooks use separate seeds for data ordering, model initialisation, diffusion timesteps and noise; evaluation uses seed 2026.
- Deterministic settings are enabled with `warn_only=True`. Exact numerical agreement across hardware and software versions is not guaranteed.
- On Windows/Jupyter, set `NUM_WORKERS = 0` if DataLoader multiprocessing fails. The loader already conditionally disables its prefetch/persistent-worker arguments when workers are zero.
- If no samples are found, check the working directory and ensure `pair_*.npz` files are directly inside the configured dataset directory.
- A missing `best.pt` means the matching model must be trained before evaluation. Training and evaluation must use the same representation and U-Net capacity.
- Some copied section headings or progress descriptions may use a generic model name. Identify the experiment from its filename, configuration and representation loader.

The repository supports rerunning the experiments from data generation through training and evaluation. It does not currently provide pretrained weights for evaluation without retraining.
