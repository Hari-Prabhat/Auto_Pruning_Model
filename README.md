# Coupling-aware gated structured pruning of EdgeNeXt

This repository holds the code for training, pruning, packaging and on-device measurement of the pruned EdgeNeXt-XXS models in the paper. It covers CIFAR-10, CIFAR-100 and Tiny-ImageNet-200, three seeds each, at 50% of the unpruned MACs, measured on a Raspberry Pi 3B+.

- **Proposed method:** hard-concrete gates on the coupled channel groups of EdgeNeXt (residual trunks, the SDTA split–cascade, XCA heads and MLP hidden units). They are trained under an augmented-Lagrangian constraint on the exact MAC count of the exported network, rounded to widths that are multiples of 8, and exported as a standard dense timm EdgeNeXt.
- **Baselines:** the unpruned model, and DepGraph (Torch-Pruning 1.6.1) restricted to the layers it can prune on EdgeNeXt. Unrestricted, it raises an `IndexError`; see the two `depgraph_*.log.txt` files.
- **Deployment:** ONNX Runtime, LiteRT FP32 and LiteRT INT8 on a Raspberry Pi 3B+.

## Repository layout

| File | Purpose |
|---|---|
| `edgenext_gated_prune.py` | Training: `baseline` (unpruned), `gated` (proposed), `depgraph` (baseline) and `check` (self-test, no data needed). Writes `model.pt` and `log.json` per run. |
| `edgenext_bench.py` | `export` packages a pruned and an unpruned model for ONNX, LiteRT FP32 and LiteRT INT8, with checksums, MACs and numerical checks. `bench` measures on a device; `report` and `summary` combine the results; `depgraph-repro` logs the DepGraph failure. |
| `run_experiment.sh` | Trains all three methods and builds the device packages for one dataset and the given seeds. |
| `get_tiny_imagenet.sh` | Downloads Tiny-ImageNet-200 into `./data`. |
| `reproduce_paper.sh` | Runs every experiment of the paper in order. |
| `edgenext_kaggle.ipynb` | The Kaggle GPU notebook used for the paper's training runs. `build_kaggle_notebook.py` rebuilds it from the two scripts. |
| `run_pi_bench.sh`, `run_pi_latency_reps.sh`, `run_int8_diag.sh`, `int8_diag.py`, `run_pi_bench_v1.sh` | Scripts run on the Raspberry Pi 3B+, with a record of each session; see `README_raspberry_pi.md`. |
| `fig3_*.py` … `fig9_*.py` | One script per result figure of the paper (Figures 3–9), drawn from the measured values. Run e.g. `python fig6_pi_speedup.py`; output goes to `figures/out/`. |
| `depgraph_edgenext_xx_small.log.txt`, `depgraph_edgenext_small.log.txt` | Torch-Pruning 1.6.1 failure logs on timm EdgeNeXt-XXS and EdgeNeXt-S. |
| `requirements-train.txt`, `requirements-export.txt`, `requirements-pi.txt` | Python packages for training, for packaging, and for the Raspberry Pi. |

`edgenext_gated_prune.py` imports `edgenext_bench.py`, so keep both in the same folder.

## Installation

```bash
# training (GPU)
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-train.txt

# packaging for the devices: a separate environment (a preinstalled TensorFlow can break litert-torch)
python -m venv .venv-export
.venv-export/bin/pip install torch==2.14.1 torchvision==0.29.1 --index-url https://download.pytorch.org/whl/cpu
.venv-export/bin/pip install -r requirements-export.txt
```

Run the self-test once on every machine. It must end with `ALL CHECKS PASSED`:

```bash
python edgenext_gated_prune.py check --archs edgenext_xx_small
```

## Running the experiments

Each command trains the unpruned baseline, the gated model and the DepGraph baseline for every seed, then builds the device packages in `artifacts/`. Runs that already finished are skipped, and interrupted training continues with `--resume`.

```bash
# CIFAR-10 and CIFAR-100 (downloaded automatically into ./data)
EXPORT_PY=.venv-export/bin/python bash run_experiment.sh cifar10  0 1 2
EXPORT_PY=.venv-export/bin/python bash run_experiment.sh cifar100 0 1 2

# Tiny-ImageNet-200
bash get_tiny_imagenet.sh
EXPORT_PY=.venv-export/bin/python DATA_ROOT=./data/tiny-imagenet-200 \
    bash run_experiment.sh tiny200 0 1 2
```

Add `DRY_RUN=1` to print the commands without running them. The underlying commands for one run (CIFAR-10, seed 0) are:

```bash
python edgenext_gated_prune.py baseline --arch edgenext_xx_small --dataset cifar10 --data-root ./data \
    --seed 0 --download --out runs/c10_xxs/baseline_s0
python edgenext_gated_prune.py gated    --arch edgenext_xx_small --dataset cifar10 --data-root ./data \
    --budget 0.5 --seed 0 --download --out runs/c10_xxs/gated_b50_s0
python edgenext_gated_prune.py depgraph --arch edgenext_xx_small --dataset cifar10 --data-root ./data \
    --budget 0.5 --seed 0 --download --out runs/c10_xxs/depgraph_b50_s0
python edgenext_bench.py export --experiment c10_xxs_gated_b50_s0 --dataset cifar10 --data-root ./data \
    --pruned runs/c10_xxs/gated_b50_s0/model.pt --unpruned runs/c10_xxs/baseline_s0/model.pt \
    --out artifacts/c10_xxs_gated_b50_s0 --download
```

Use `--dataset cifar100` (folder prefix `c100_xxs`) or `--dataset tiny200 --data-root ./data/tiny-imagenet-200` (prefix `t200_xxs`) for the other datasets. All hyperparameters come from the `xxs` recipe in `edgenext_gated_prune.py`:

- **Phase A:** 15 epochs, Adam.
- **Fine-tuning:** 10 epochs, SGD with cosine decay.
- **Batch size:** 32 for CIFAR, 64 for Tiny-ImageNet-200.

The paper's runs were made with `edgenext_kaggle.ipynb` on a Kaggle GPU. To use it, upload it to Kaggle, set `DATASET` and `SEEDS` in cell 1, turn on the GPU and internet, then choose *Save & Run All*. It runs the same commands as `run_experiment.sh`.

## Measuring on the Raspberry Pi

Copy `artifacts/*.zip` to the Pi and follow `README_raspberry_pi.md`. When the result zips come back, unpack each package's `results/` folder into `artifacts/<package>/results/`, then combine everything:

```bash
python edgenext_bench.py report  --results "artifacts/c10_xxs_gated_b50_s0/results/*.json" --out reports/c10_xxs_gated_b50_s0
python edgenext_bench.py summary --logs "runs/**/log.json" --results "artifacts/*/results/*.json" --out reports/summary
```

## DepGraph failure evidence

```bash
python edgenext_bench.py depgraph-repro --arch edgenext_xx_small --out evidence/rerun/depgraph_xxs.log
python edgenext_bench.py depgraph-repro --arch edgenext_xx_small --patch-xca --out evidence/rerun/depgraph_xxs_xcapatch.log
python edgenext_bench.py depgraph-repro --arch edgenext_small --out evidence/rerun/depgraph_s.log
```

The two `depgraph_*.log.txt` logs were written by an earlier revision of `edgenext_bench.py`, so the line numbers in their tracebacks differ from the current file. The error comes from inside Torch-Pruning and is the same: `index 48 is out of bounds for dimension 0 with size 48` for XXS, and 96 for S.

## Software versions used in the paper

| Step | Versions |
|---|---|
| Training | PyTorch 2.11.0 (CUDA 12.8), timm 1.0.30, Torch-Pruning 1.6.1, one Kaggle GPU |
| Export | PyTorch 2.14.1 (CPU), timm 1.0.30, ONNX opset 17, onnxruntime 1.29.0, litert-torch 0.10.0, ai-edge-quantizer 0.10.0, ai-edge-litert 2.3.0 |
| Raspberry Pi 3B+ | 64-bit Raspberry Pi OS, Python 3.13.5, numpy 2.5.4, onnxruntime 1.31.0, ai-edge-litert 2.3.0, 4 threads |

## Notes on provenance

- **Earliest runs:** the first runs (CIFAR-10 seed 0, and the first CIFAR-100 baseline and gated runs) were trained with an earlier copy of `edgenext_gated_prune.py`. That copy differs only in the safety check that stops a fine-tuning run collapsing to chance level, plus one extra log line. The old check compared against the phase-A loss and wrongly stopped the CIFAR-100 DepGraph runs; those were rerun with the file in this repository. Training and pruning are otherwise identical.
- **Datasets, trained models and packages** are not included because they are large; `.gitignore` excludes them.

## Citation

Y. S. Kumari, Hari Prabhat Kumar, M. Yagneshwar, K. M. Shanthan, "Coupling-Aware Gated Structured Pruning of EdgeNeXt for Deployment on Low-Cost Edge Devices", submitted to the Journal of Systems Architecture, 2026. The full reference will be added once the paper is published.

## License

MIT License; see `LICENSE`.
