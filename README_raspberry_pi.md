# Raspberry Pi 3B+ measurements

These are the scripts that were run on the Raspberry Pi 3 Model B+ (64-bit Raspberry Pi OS, no heatsink or fan) for the paper. Each one measures the device packages built by `edgenext_bench.py export` (see `run_experiment.sh`). Each package holds the pruned and the unpruned model in three formats: ONNX, LiteRT FP32 and LiteRT INT8. It also contains the evaluation images, a manifest with checksums, and its own copy of `edgenext_bench.py`.

| Script | What it measures | Output to send back |
|---|---|---|
| `run_pi_bench.sh` | Accuracy on about 1,000 class-balanced test images, batch-1 latency, 60 s sustained throughput, peak RAM, model load time and first-inference time. Each runtime is measured separately, after the CPU has cooled below 50 °C. | `pi_results_<label>_<date>.zip` |
| `run_pi_latency_reps.sh` | Latency only, in 5 independent sessions per package and runtime, each started below 50 °C. The paper reports the median of these 5 sessions. | `pi_latreps_<label>_<date>.zip` |
| `run_int8_diag.sh` (with `int8_diag.py`) | Runs the same CIFAR-100 INT8 models with XNNPACK (4 and 1 threads), the built-in kernels and the TFLite reference kernels, so the Pi's predictions can be compared with a PC. | `pi_int8diag_<date>.zip` |
| `run_pi_bench_v1.sh` | First version of `run_pi_bench.sh`, which measured all runtimes in one go without cool-down. It was used only for the first CIFAR-10 seed-0 session; `run_pi_bench.sh` moves results of this version to `results_superseded/`. | `pi_results_<label>_<date>.zip` |

## How to run

1. Copy the package zips (e.g. `c10_xxs_gated_b50_s0.zip`, `c10_xxs_depgraph_b50_s0.zip`) and `run_pi_bench.sh` into one folder on the Pi, e.g. `~/benchmark`.
2. Run `cd ~/benchmark && bash run_pi_bench.sh`. Over SSH, start it inside `tmux` so it keeps running if the connection drops. The first run creates `~/benchenv` and installs numpy, pillow, onnxruntime and ai-edge-litert.
3. Then run `bash run_pi_latency_reps.sh ~/benchmark` for the repeated latency sessions. It can be stopped and restarted; finished sessions are skipped.
4. Copy the result zips back, unpack them into `artifacts/<package>/`, and run `edgenext_bench.py report` / `summary` (see the main README).

Optional settings, such as `LABEL`, `SUSTAINED`, `COOL_C`, `REPS` and `RERUN=1`, are described at the top of each script.

## Sessions run for the paper

All sessions used device label `rpi3bplus`, Python 3.13.5, numpy 2.5.4, ONNX Runtime 1.31.0 and ai-edge-litert 2.3.0, with 4 threads.

| Date (IST) | Script | Packages |
|---|---|---|
| 25 Oct 2026, 14:08 | `run_pi_bench_v1.sh` | CIFAR-10 seed 0 (gated, DepGraph) |
| 25 Oct 2026, 18:56 | `run_pi_bench.sh` | CIFAR-10 seeds 1–2 (gated, DepGraph) |
| 26 Oct 2026, 07:15 | `run_pi_latency_reps.sh` | CIFAR-10 seeds 1–2 |
| 26 Oct 2026, 09:21 | `run_pi_bench.sh` | CIFAR-100 seeds 0–2 |
| 26 Oct 2026, 11:31 | `run_pi_latency_reps.sh` | CIFAR-100 seeds 0–2 |
| 26 Oct 2026, 16:43 | `run_pi_bench.sh` | Tiny-ImageNet-200 seeds 0–2 |
| 26 Oct 2026, 18:47 | `run_pi_latency_reps.sh` | Tiny-ImageNet-200 seeds 0–2 |
| 27 Oct 2026, 19:33 | `run_pi_latency_reps.sh` | CIFAR-10 seed 0 |

The INT8 diagnostic kit also needs three optional `*_int8_allout.tflite` files, which expose every intermediate tensor. They are model files and are not stored in this repository. Without them the kit still compares the predictions of all four kernel modes and skips only the per-tensor dump.
