"""
edgenext_bench.py - pruned vs. unpruned EdgeNeXt (timm), measured end to end.

Every number this script writes is measured in the run that writes it. Nothing
is typed in, estimated, copied from another run or generated at random. Each
results file names the device, runtime versions, thread count, the exact model
files (SHA-256) and the exact evaluation images it used, so every figure in a
report can be traced back to a measurement.

Subcommands
-----------
  export           (laptop, needs torch + timm) Package one experiment: both
                   models as PyTorch / ONNX / TFLite FP32 / TFLite INT8, the
                   evaluation images, and a manifest with checksums, parameter
                   counts and MACs (all computed, never typed).
  bench            (any device) Measure accuracy and latency of pruned AND
                   unpruned on every runtime available on this device. Verifies
                   file checksums against the manifest first.
  report           Combine results files from all devices into one Markdown /
                   CSV report with speedups, confidence intervals and
                   significance tests computed from the raw measurements.
  summary          Every run and seed at once (training log.json files +
                   results files): accuracy mean +- std, paired change vs the
                   baseline, complexity, training time, device metrics and
                   accuracy-vs-cost figures.
  depgraph-repro   Run Torch-Pruning (DepGraph) on timm EdgeNeXt and log what
                   happens (traceback, versions), as evidence for the paper.

Metrics recorded
----------------
  export   params, MACs/FLOPs, file size per format, ONNX/TFLite numerical
           checks, INT8 checks, TFLite weight and peak activation memory
           (tensor-arena estimate for microcontrollers)
  bench    top-1 and top-5 with 95% intervals, per-image predictions,
           batch-1 latency (mean, std, median, p5, p95, p99, min, max,
           throughput), model load time, first-inference (cold-start)
           latency, peak RAM in a fresh process, CPU temperature, frequency
           and Raspberry Pi throttling flags; with --sustained-seconds:
           sustained throughput and energy per inference (on-board sensor,
           or an external USB meter log given to report/summary --power-csv)
  report   speedup with bootstrap CI, McNemar test, runtime agreement

Training (unpruned baseline, gated pruning, DepGraph baseline) lives in
edgenext_gated_prune.py, which imports this file; keep both in one folder.

Typical order for one experiment (example: CIFAR-10, EdgeNeXt-XXS, seed 0)
-------------------------------------------------------------------------
  # 1. train on a GPU (edgenext_gated_prune.py)
  python edgenext_gated_prune.py baseline --arch edgenext_xx_small --dataset cifar10 \
      --data-root ./data --recipe xxs --seed 0 --out runs/c10_xxs/baseline_s0
  python edgenext_gated_prune.py gated --arch edgenext_xx_small --dataset cifar10 \
      --data-root ./data --recipe xxs --budget 0.5 --seed 0 --out runs/c10_xxs/gated_b50_s0

  # 2. package both models (laptop or the GPU machine)
  python edgenext_bench.py export --experiment c10_xxs_b50_s0 --dataset cifar10 \
      --data-root ./data --pruned runs/c10_xxs/gated_b50_s0/model.pt \
      --unpruned runs/c10_xxs/baseline_s0/model.pt --out artifacts/c10_xxs_b50_s0

  # 3. measure on every device (copy the artifacts folder; the script is inside)
  python edgenext_bench.py bench --artifacts artifacts/c10_xxs_b50_s0 --device-label laptop \
      --sustained-seconds 60
  python3 edgenext_bench.py bench --artifacts c10_xxs_b50_s0 --device-label rpi3bplus
  python3 edgenext_bench.py bench --artifacts c10_xxs_b50_s0 --device-label qemu-a72 --emulated yes

  # 4. one report per package, then one summary of every run
  python edgenext_bench.py report --results artifacts/c10_xxs_b50_s0/results/*.json \
      --out reports/c10_xxs_b50_s0
  python edgenext_bench.py summary --logs "runs/**/log.json" \
      --results "artifacts/*/results/*.json" --out reports/summary

Device-side requirements: numpy, pillow, plus onnxruntime and/or ai-edge-litert
(or tflite-runtime). torch/timm are only needed for the "pytorch" runtime,
export and depgraph-repro.
"""

import argparse
import csv
import glob
import hashlib
import json
import math
import os
import pickle
import platform
import shutil
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone

import numpy as np

SCHEMA = "edgenext-bench/1"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

TINY10_DEFAULT_WNIDS = [
    "n02124075", "n04067472", "n04540053", "n04099969", "n07749582",
    "n01641577", "n02802426", "n09246464", "n07920052", "n03970156",
]

DATASETS = {
    
    "cifar10": {"num_classes": 10, "eval_split": "test",
                "preprocess": {"mode": "resize", "size": 224}},
    "cifar100": {"num_classes": 100, "eval_split": "test",
                 "preprocess": {"mode": "resize", "size": 224}},
    "tiny10": {"num_classes": 10, "eval_split": "val",
               "preprocess": {"mode": "resize_short_center_crop", "resize": 256, "size": 224}},
    "tiny200": {"num_classes": 200, "eval_split": "val",
                "preprocess": {"mode": "resize_short_center_crop", "resize": 256, "size": 224}},
    "imagefolder": {"num_classes": None, "eval_split": "val",
                    "preprocess": {"mode": "presized", "resize": 256, "size": 224}},
}
IMG_EXTS = (".jpeg", ".jpg", ".png", ".bmp", ".webp")

ARCH_BY_STAGE4_DIM = {168: "edgenext_xx_small", 192: "edgenext_x_small",
                      304: "edgenext_small", 584: "edgenext_base"}

ALL_RUNTIMES = ["pytorch", "onnx", "tflite_fp32", "tflite_int8"]



def now_iso():
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def file_record(path, base_dir):
    return {"file": os.path.relpath(path, base_dir), "sha256": sha256_file(path),
            "bytes": os.path.getsize(path)}


def write_json(path, obj):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def read_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def log(msg):
    print(msg, flush=True)


def pkg_version(name):
    try:
        from importlib.metadata import version
        return version(name)
    except Exception:
        return None


def _read_text(path):
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    except Exception:
        return ""


def cpu_model_name():
    info = _read_text("/proc/cpuinfo")
    for key in ("model name", "Model", "Hardware", "CPU part"):
        for line in info.splitlines():
            if line.split(":")[0].strip() == key:
                return line.split(":", 1)[1].strip()
    if platform.system() == "Windows":
        return platform.processor() or None
    if platform.system() == "Darwin":
        try:
            return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                  capture_output=True, text=True, timeout=5).stdout.strip()
        except Exception:
            return None
    return platform.processor() or None


def looks_emulated(hints):
    """True only for instruction-set emulation (QEMU TCG / qemu-user). WSL, KVM and
    Hyper-V are virtualised but run native instructions, so they do not count."""
    return any("qemu" in h.lower() for h in hints)


def detect_emulation():
    """Best-effort hints only; the --emulated flag is authoritative."""
    hints = []
    try:
        r = subprocess.run(["systemd-detect-virt"], capture_output=True, text=True, timeout=5)
        v = r.stdout.strip()
        if v and v != "none":
            hints.append("systemd-detect-virt=" + v)
    except Exception:
        pass
    vendor = _read_text("/sys/class/dmi/id/sys_vendor").strip()
    if "QEMU" in vendor:
        hints.append("dmi sys_vendor=" + vendor)
    cpuinfo = _read_text("/proc/cpuinfo")
    if platform.machine() in ("aarch64", "arm64") and ("GenuineIntel" in cpuinfo or "AuthenticAMD" in cpuinfo):
        hints.append("aarch64 userland on x86 cpuinfo (qemu-user / binfmt)")
    if platform.machine() in ("aarch64", "arm64") and "QEMU" in cpuinfo:
        hints.append("qemu in /proc/cpuinfo")
    if os.environ.get("QEMU_CPU"):
        hints.append("QEMU_CPU=" + os.environ["QEMU_CPU"])
    return hints


def env_info():
    return {
        "hostname": socket.gethostname(),
        "os": platform.platform(),
        "machine": platform.machine(),
        "cpu_model": cpu_model_name(),
        "board_model": _read_text("/proc/device-tree/model").strip("\x00\n ") or None,
        "logical_cpus": os.cpu_count(),
        "python": platform.python_version(),
        "packages": {p: pkg_version(p) for p in (
            "numpy", "pillow", "torch", "timm", "onnx", "onnxruntime", "ai-edge-litert",
            "tflite-runtime", "tensorflow", "litert-torch", "ai-edge-torch",
            "ai-edge-quantizer", "torch-pruning")},
        "emulation_hints": detect_emulation(),
    }



def _proc_status_kib(key):
    for line in _read_text("/proc/self/status").splitlines():
        if line.startswith(key + ":"):
            try:
                return int(line.split()[1])
            except (IndexError, ValueError):
                return None
    return None


def rss_bytes():
    """Current resident memory of this process."""
    v = _proc_status_kib("VmRSS")
    if v is not None:
        return v * 1024
    try:
        import psutil
        return int(psutil.Process().memory_info().rss)
    except Exception:
        return None


def peak_rss_bytes():
    """Peak resident memory of this process since it started."""
    v = _proc_status_kib("VmHWM")
    if v is not None:
        return v * 1024
    try:
        import resource
        r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return int(r if sys.platform == "darwin" else r * 1024)
    except Exception:
        pass
    try:
        import psutil
        return int(getattr(psutil.Process().memory_info(), "peak_wset", 0)) or None
    except Exception:
        return None


_THROTTLE_BITS = {0: "under_voltage_now", 1: "freq_capped_now", 2: "throttled_now", 3: "soft_temp_limit_now",
                  16: "under_voltage_since_boot", 17: "freq_capped_since_boot", 18: "throttled_since_boot",
                  19: "soft_temp_limit_since_boot"}


def device_state():
    """CPU temperatures (sysfs thermal zones), current CPU frequencies and the
    Raspberry Pi throttling flags (vcgencmd get_throttled)."""
    st = {"unix_time": time.time(), "temps_c": {}, "max_temp_c": None, "cpu_mhz": None, "throttled": None}
    for z in sorted(glob.glob("/sys/class/thermal/thermal_zone*")):
        t = _read_text(os.path.join(z, "temp")).strip()
        if t.lstrip("-").isdigit():
            name = _read_text(os.path.join(z, "type")).strip() or os.path.basename(z)
            st["temps_c"][name] = int(t) / 1000.0
    if st["temps_c"]:
        st["max_temp_c"] = max(st["temps_c"].values())
    f = []
    for c in sorted(glob.glob("/sys/devices/system/cpu/cpu[0-9]*/cpufreq/scaling_cur_freq")):
        v = _read_text(c).strip()
        if v.isdigit():
            f.append(int(v) / 1000.0)
    st["cpu_mhz"] = f or None
    if shutil.which("vcgencmd"):
        try:
            out = subprocess.run(["vcgencmd", "get_throttled"], capture_output=True, text=True, timeout=3).stdout
            v = int(out.strip().split("=")[1], 16)
            st["throttled"] = dict({"raw": hex(v)}, **{n: bool(v >> b & 1) for b, n in _THROTTLE_BITS.items()})
        except Exception:
            pass
    return st


class PowerMeter:
    """Energy over a time window from an on-board sensor, if this device has one:
      rapl      Intel/AMD CPU package energy counters (native Linux; usually needs
                root or `sudo chmod a+r /sys/class/powercap/intel-rapl:*/energy_uj`)
      pi5_pmic  Raspberry Pi 5 PMIC rails via `vcgencmd pmic_read_adc`, sampled
                (SoC and memory rails; USB and other peripherals not included)
      ina3221   Jetson VDD_IN input-rail monitor (hwmon), sampled
    Raspberry Pi 3B+/4 and most laptops under WSL2 have none: use an external
    USB power meter with the window timestamps (see bench --sustained-seconds)."""

    def __init__(self):
        self.kind, self.detail, self.reason = None, None, "no on-board power sensor found"
        self._rapl, self._hw = [], None
        for d in sorted(glob.glob("/sys/class/powercap/intel-rapl:*")):
            if os.path.basename(d).count(":") != 1:
                continue
            e = os.path.join(d, "energy_uj")
            if not os.path.exists(e):
                continue
            if not os.access(e, os.R_OK) or not _read_text(e).strip().isdigit():
                self.reason = "RAPL present but %s is not readable (run as root or chmod a+r)" % e
                continue
            rng = _read_text(os.path.join(d, "max_energy_range_uj")).strip()
            self._rapl.append((e, int(rng) if rng.isdigit() else None,
                               _read_text(os.path.join(d, "name")).strip()))
        if self._rapl:
            self.kind = "rapl"
            self.detail = "CPU package energy counters: " + ", ".join(n for _, _, n in self._rapl)
            self.reason = None
            return
        if "Raspberry Pi 5" in _read_text("/proc/device-tree/model") and self._pi5_power() is not None:
            self.kind, self.detail, self.reason = "pi5_pmic", "Pi 5 PMIC rails (SoC + memory), sampled 5 Hz", None
            return
        for lbl in glob.glob("/sys/bus/i2c/drivers/ina3221*/*/hwmon/hwmon*/in*_label"):
            if "VDD_IN" in _read_text(lbl):
                n = os.path.basename(lbl)[2:-6]
                base = os.path.dirname(lbl)
                self._hw = (os.path.join(base, "in%s_input" % n), os.path.join(base, "curr%s_input" % n))
                if self._ina_power() is not None:
                    self.kind, self.detail, self.reason = "ina3221", "Jetson VDD_IN rail, sampled 5 Hz", None
                    return

    @staticmethod
    def _pi5_power():
        try:
            out = subprocess.run(["vcgencmd", "pmic_read_adc"], capture_output=True, text=True, timeout=3).stdout
        except Exception:
            return None
        amps, volts = {}, {}
        for line in out.splitlines():
            parts = line.split()
            if len(parts) < 2 or "=" not in parts[1]:
                continue
            name, val = parts[0], parts[1].split("=")[1]
            try:
                x = float(val.rstrip("AV"))
            except ValueError:
                continue
            if name.endswith("_A"):
                amps[name[:-2]] = x
            elif name.endswith("_V"):
                volts[name[:-2]] = x
        common = set(amps) & set(volts)
        return sum(amps[k] * volts[k] for k in common) if common else None

    def _ina_power(self):
        v, i = _read_text(self._hw[0]).strip(), _read_text(self._hw[1]).strip()
        if v.isdigit() and i.isdigit():
            return int(v) * int(i) / 1e6
        return None

    def start(self):
        if self.kind is None:
            return None
        if self.kind == "rapl":
            return {"t": time.perf_counter(), "e": [int(_read_text(p).strip()) for p, _, _ in self._rapl]}
        import threading
        tok = {"t": time.perf_counter(), "j": 0.0, "n": 0, "stop": threading.Event()}
        fn = self._pi5_power if self.kind == "pi5_pmic" else self._ina_power

        def loop():
            t_prev, p_prev = time.perf_counter(), fn()
            while not tok["stop"].wait(0.2):
                t, p = time.perf_counter(), fn()
                if p is not None and p_prev is not None:
                    tok["j"] += 0.5 * (p + p_prev) * (t - t_prev)
                    tok["n"] += 1
                t_prev, p_prev = t, p
        tok["thread"] = threading.Thread(target=loop, daemon=True)
        tok["thread"].start()
        return tok

    def stop(self, tok):
        if tok is None:
            return None
        if self.kind == "rapl":
            dt = time.perf_counter() - tok["t"]
            j = 0.0
            for (p, rng, _), e0 in zip(self._rapl, tok["e"]):
                d = int(_read_text(p).strip()) - e0
                if d < 0 and rng:
                    d += rng
                j += d / 1e6
            return {"joules": j, "seconds": dt, "avg_w": j / dt if dt > 0 else None, "samples": None}
        tok["stop"].set()
        tok["thread"].join()
        dt = time.perf_counter() - tok["t"]
        return {"joules": tok["j"], "seconds": dt, "avg_w": tok["j"] / dt if dt > 0 else None,
                "samples": tok["n"]}


def tflite_memory_estimate(path):
    """Weights and activation memory of a .tflite graph, from its tensors and the
    op execution order: peak = largest total size of activation tensors alive
    at the same time (no scratch buffers, no allocator overhead). An estimate of
    the TFLite Micro tensor arena; the firmware's arena_used_bytes() is the
    measured value."""
    Interpreter, _ = _tflite_interpreter_cls()
    kw = {}
    try:
        from ai_edge_litert.interpreter import OpResolverType
        kw["experimental_op_resolver_type"] = OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES
    except Exception:
        pass
    try:
        it = Interpreter(model_path=path, **kw)
    except TypeError:
        it = Interpreter(model_path=path)
    it.allocate_tensors()
    td = {t["index"]: t for t in it.get_tensor_details()}
    ops = it._get_ops_details()

    def nbytes(i):
        t = td.get(i)
        if t is None:
            return 0
        return int(np.prod(t["shape"])) * np.dtype(t["dtype"]).itemsize if len(t["shape"]) else \
            np.dtype(t["dtype"]).itemsize

    produced = {}
    last_use = {}
    for k, op in enumerate(ops):
        for i in op["inputs"]:
            if i >= 0:
                last_use[i] = k
        for o in op["outputs"]:
            produced[o] = k
    inputs = [d["index"] for d in it.get_input_details()]
    outputs = [d["index"] for d in it.get_output_details()]
    live = {}
    for i in inputs:
        live[i] = (-1, last_use.get(i, -1))
    for o, k in produced.items():
        live[o] = (k, len(ops) if o in outputs else max(last_use.get(o, k), k))
    peak, at = 0, None
    for k in range(len(ops)):
        s = sum(nbytes(i) for i, (a, b) in live.items() if a <= k <= b)
        if s > peak:
            peak, at = s, k
    weights = sum(nbytes(i) for i in last_use if i not in live)
    return {"peak_activation_bytes": int(peak), "peak_at_op": ops[at]["op_name"] if at is not None else None,
            "largest_activation_bytes": int(max((nbytes(i) for i in live), default=0)),
            "activation_tensors_total_bytes": int(sum(nbytes(i) for i in live)),
            "weights_bytes": int(weights), "ops": len(ops),
            "note": "liveness estimate over the op order; excludes scratch buffers and allocator overhead"}



def latency_summary(ms):
    a = np.asarray(ms, dtype=np.float64)
    return {
        "n": int(a.size), "mean": float(a.mean()), "std": float(a.std(ddof=1)) if a.size > 1 else 0.0,
        "median": float(np.median(a)), "p05": float(np.percentile(a, 5)),
        "p95": float(np.percentile(a, 95)), "p99": float(np.percentile(a, 99)),
        "min": float(a.min()), "max": float(a.max()),
        "throughput_img_s_from_mean": float(1000.0 / a.mean()),
        "throughput_img_s_from_median": float(1000.0 / np.median(a)),
    }


def wilson_interval(correct, n, z=1.959963984540054):
    if n == 0:
        return [float("nan"), float("nan")]
    p = correct / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [max(0.0, centre - half), min(1.0, centre + half)]


def mcnemar_exact(correct_a, correct_b):
    """Two-sided exact McNemar test on paired per-image correctness."""
    a = np.asarray(correct_a, bool)
    b = np.asarray(correct_b, bool)
    n01 = int(np.sum(a & ~b))   
    n10 = int(np.sum(~a & b))   
    n = n01 + n10
    if n == 0:
        return {"a_only_correct": n01, "b_only_correct": n10, "p_value": 1.0}
    k = min(n01, n10)
    logs = [math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) - n * math.log(2)
            for i in range(k + 1)]
    m = max(logs)
    tail = math.exp(m) * sum(math.exp(x - m) for x in logs)
    return {"a_only_correct": n01, "b_only_correct": n10, "p_value": min(1.0, 2.0 * tail)}


def bootstrap_ratio_of_medians(num_ms, den_ms, iters=4000, seed=0):
    """CI for median(num)/median(den), resampling each sample independently."""
    rng = np.random.default_rng(seed)
    a = np.asarray(num_ms, np.float64)
    b = np.asarray(den_ms, np.float64)
    ra = np.median(a[rng.integers(0, a.size, (iters, a.size))], axis=1)
    rb = np.median(b[rng.integers(0, b.size, (iters, b.size))], axis=1)
    r = ra / rb
    return [float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))]



def _cifar_files(root, dataset, split):
    if dataset == "cifar10":
        base = os.path.join(root, "cifar-10-batches-py")
        names = ["test_batch"] if split == "test" else ["data_batch_%d" % i for i in range(1, 6)]
        key = b"labels"
    else:
        base = os.path.join(root, "cifar-100-python")
        names = ["test"] if split == "test" else ["train"]
        key = b"fine_labels"
    return [os.path.join(base, n) for n in names], key


def imagefolder_classes(root):
    d = os.path.join(root, "train")
    if not os.path.isdir(d):
        raise FileNotFoundError("expected %s/train/<class>/... and %s/val/<class>/..." % (root, root))
    return sorted(x for x in os.listdir(d) if os.path.isdir(os.path.join(d, x)))


def list_imagefolder(root, split, per_class=None):
    """(paths, labels, classes) for <root>/<split>/<class>/<image>; class order = sorted train folders."""
    classes = imagefolder_classes(root)
    paths, labels = [], []
    for label, c in enumerate(classes):
        folder = os.path.join(root, split, c)
        if not os.path.isdir(folder):
            raise FileNotFoundError("missing %s" % folder)
        files = sorted(f for f in os.listdir(folder) if f.lower().endswith(IMG_EXTS))
        if per_class is not None:
            files = files[:per_class]
        paths += [os.path.join(folder, f) for f in files]
        labels += [label] * len(files)
    return paths, np.asarray(labels, np.int64), classes


def dataset_num_classes(dataset, root):
    n = DATASETS[dataset]["num_classes"]
    return n if n is not None else len(imagefolder_classes(root))


def _resize_short_center_crop(im, resize, size):
    """torchvision Resize(int) + CenterCrop(size) semantics on a PIL image (bilinear)."""
    from PIL import Image
    bilinear = getattr(Image, "Resampling", Image).BILINEAR
    w, h = im.size
    if w <= h:
        ow, oh = resize, int(resize * h / w)
    else:
        oh, ow = resize, int(resize * w / h)
    if (ow, oh) != (w, h):
        im = im.resize((ow, oh), bilinear)
    top = int(round((oh - size) / 2.0))
    left = int(round((ow - size) / 2.0))
    return im.crop((left, top, left + size, top + size))


def load_presized(path, cfg):
    """Open an image file and apply the fixed resize + centre-crop (uint8 HWC)."""
    from PIL import Image
    with Image.open(path) as im:
        return np.asarray(_resize_short_center_crop(im.convert("RGB"), int(cfg["resize"]), int(cfg["size"])),
                          np.uint8)


def load_raw_split(dataset, root, split, class_order=None, download=False, per_class=None):
    """Returns (images uint8 [N,H,W,3], labels int64 [N], class_names list).
    For imagefolder the images are already resized/cropped (they can be large),
    and per_class caps how many are loaded."""
    if dataset == "imagefolder":
        cfg = DATASETS["imagefolder"]["preprocess"]
        paths, labels, classes = list_imagefolder(root, split, per_class)
        if not paths:
            raise FileNotFoundError("no images under %s/%s" % (root, split))
        return np.stack([load_presized(p, cfg) for p in paths]), labels, classes
    if dataset in ("cifar10", "cifar100"):
        files, key = _cifar_files(root, dataset, split)
        if not all(os.path.exists(f) for f in files):
            if download:
                import torchvision
                cls = torchvision.datasets.CIFAR10 if dataset == "cifar10" else torchvision.datasets.CIFAR100
                cls(root=root, train=(split == "train"), download=True)
            else:
                raise FileNotFoundError(
                    "CIFAR files not found under %s (expected %s). Pass --download or point "
                    "--data-root at the folder that contains them." % (root, files[0]))
        imgs, labels = [], []
        for f in files:
            with open(f, "rb") as fh:
                d = pickle.load(fh, encoding="bytes")
            imgs.append(np.asarray(d[b"data"], np.uint8).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1))
            labels.extend(d[key])
        meta = os.path.join(os.path.dirname(files[0]), "batches.meta" if dataset == "cifar10" else "meta")
        names = None
        if os.path.exists(meta):
            with open(meta, "rb") as fh:
                m = pickle.load(fh, encoding="bytes")
            raw = m.get(b"label_names") or m.get(b"fine_label_names")
            names = [x.decode() for x in raw] if raw else None
        imgs, labels = np.concatenate(imgs), np.asarray(labels, np.int64)
        if per_class is not None:
            keep = stratified_indices(labels, per_class=per_class, num_classes=DATASETS[dataset]["num_classes"])
            imgs, labels = imgs[keep], labels[keep]
        return np.ascontiguousarray(imgs), labels, names

    if dataset in ("tiny10", "tiny200"):
        from PIL import Image
        n_cls = DATASETS[dataset]["num_classes"]
        wnids = list(class_order) if class_order else None
        if wnids is None:
            wf = os.path.join(root, "wnids.txt")
            if os.path.exists(wf):
                with open(wf) as fh:
                    wnids = [l.strip() for l in fh if l.strip()][:n_cls]
            elif dataset == "tiny10":
                wnids = list(TINY10_DEFAULT_WNIDS)
            else:
                wnids = sorted(os.listdir(os.path.join(root, "train")))[:n_cls]
        if len(wnids) != n_cls:
            raise ValueError("%s needs %d classes, found %d" % (dataset, n_cls, len(wnids)))
        paths, labels = [], []
        for label, wnid in enumerate(wnids):
            if split == "train":
                cands = [os.path.join(root, "train", wnid, "images"), os.path.join(root, "train", wnid)]
                folder = next((c for c in cands if os.path.isdir(c)), None)
                if folder is None:
                    raise FileNotFoundError("No train folder for %s under %s" % (wnid, root))
                files = sorted(f for f in os.listdir(folder) if f.lower().endswith((".jpeg", ".jpg", ".png")))
                paths += [os.path.join(folder, f) for f in files]
                labels += [label] * len(files)
        if split == "val":
            ann = os.path.join(root, "val", "val_annotations.txt")
            if os.path.exists(ann):
                with open(ann) as fh:
                    rows = [l.split() for l in fh if l.strip()]
                idx = {w: i for i, w in enumerate(wnids)}
                rows = sorted((r[1], r[0]) for r in rows if r[1] in idx)
                for wnid, fn in rows:
                    paths.append(os.path.join(root, "val", "images", fn))
                    labels.append(idx[wnid])
            else:
                for label, wnid in enumerate(wnids):
                    folder = os.path.join(root, "val", wnid)
                    if not os.path.isdir(folder):
                        raise FileNotFoundError("No val folder for %s under %s" % (wnid, root))
                    files = sorted(f for f in os.listdir(folder) if f.lower().endswith((".jpeg", ".jpg", ".png")))
                    paths += [os.path.join(folder, f) for f in files]
                    labels += [label] * len(files)
        if not paths:
            raise FileNotFoundError("No %s images found for %s under %s" % (split, dataset, root))
        labels = np.asarray(labels, np.int64)
        if per_class is not None:
            keep = stratified_indices(labels, per_class=per_class, num_classes=n_cls)
            paths, labels = [paths[i] for i in keep], labels[keep]
        imgs = np.stack([np.asarray(Image.open(p).convert("RGB"), np.uint8) for p in paths])
        return imgs, labels, wnids

    raise ValueError("unknown dataset " + dataset)


def stratified_indices(labels, per_class=None, total=None, num_classes=None, spread=False):
    """Deterministic class-balanced subset. per_class wins over total.
    spread=False: first k of each class (in file order); spread=True: evenly spaced."""
    labels = np.asarray(labels)
    nc = num_classes or int(labels.max()) + 1
    if per_class is None:
        if total is None:
            return np.arange(labels.size)
        per_class = max(1, int(math.ceil(total / nc)))
    out = []
    for c in range(nc):
        idx = np.flatnonzero(labels == c)
        if idx.size == 0:
            continue
        if spread and idx.size > per_class:
            idx = idx[np.linspace(0, idx.size - 1, per_class).round().astype(int)]
        out.append(idx[:per_class])
    return np.sort(np.concatenate(out))


def preprocess(img_u8, cfg):
    """uint8 HWC -> float32 CHW, identical to the pruning scripts' eval transform
    (torchvision Resize on PIL images = PIL bilinear; ToTensor; ImageNet Normalize)."""
    from PIL import Image
    bilinear = getattr(Image, "Resampling", Image).BILINEAR
    size = int(cfg["size"])
    if cfg["mode"] == "presized":
        if img_u8.shape[:2] != (size, size):
            raise ValueError("presized image has shape %s, expected %dx%d" % (img_u8.shape, size, size))
        x = np.asarray(img_u8, np.float32) / 255.0
    else:
        im = Image.fromarray(img_u8)
        if cfg["mode"] == "resize":
            im = im.resize((size, size), bilinear)
        elif cfg["mode"] == "resize_short_center_crop":
            im = _resize_short_center_crop(im, int(cfg["resize"]), size)
        else:
            raise ValueError(cfg["mode"])
        x = np.asarray(im, np.float32) / 255.0
    x = (x - np.asarray(IMAGENET_MEAN, np.float32)) / np.asarray(IMAGENET_STD, np.float32)
    return np.ascontiguousarray(x.transpose(2, 0, 1))


def preprocess_batch(imgs_u8, cfg):
    return np.stack([preprocess(im, cfg) for im in imgs_u8]).astype(np.float32)



def install_xca_patch():
    """timm's CrossCovarianceAttn reshapes to the INPUT width after attention,
    which breaks once heads are pruned. The pruned checkpoints were pickled with
    instance forwards bound to `CrossCovarianceAttn.patched_forward`, so that
    attribute must exist before torch.load. For unpruned models the patched
    forward is numerically identical to timm's."""
    import torch.nn.functional as F
    from timm.models.edgenext import CrossCovarianceAttn

    def patched_forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 4, 1)
        q, k, v = qkv.unbind(0)
        attn = (F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).permute(0, 3, 1, 2).reshape(B, N, -1)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    CrossCovarianceAttn.patched_forward = patched_forward
    CrossCovarianceAttn.forward = patched_forward


def install_sdta_patch():
    """timm's SplitTransposeBlock splits its input with x.chunk(n_convs + 1),
    which only works for the original widths. This forward splits with explicit
    sizes [w] * n_convs + [C - n_convs * w], w = width of the depthwise convs,
    so a block whose trunk was pruned (smaller w and/or smaller pass-through
    part) still runs. For unpruned models the sizes equal chunk()'s sizes, so
    the output is identical."""
    import torch
    from timm.models.edgenext import SplitTransposeBlock

    def split_forward(self, x):
        shortcut = x
        n = len(self.convs)
        w = int(self.convs[0].in_channels)
        c = int(self.norm_xca.normalized_shape[0])
        spx = torch.split(x, [w] * n + [c - w * n], dim=1)
        spo = []
        sp = spx[0]
        for i, conv in enumerate(self.convs):
            if i > 0:
                sp = sp + spx[i]
            sp = conv(sp)
            spo.append(sp)
        spo.append(spx[-1])
        x = torch.cat(spo, 1)

        B, C, H, W = x.shape
        x = x.reshape(B, C, H * W).permute(0, 2, 1)
        if self.pos_embd is not None:
            pos_encoding = self.pos_embd((B, H, W)).reshape(B, -1, x.shape[1]).permute(0, 2, 1)
            x = x + pos_encoding
        x = x + self.drop_path(self.gamma_xca * self.xca(self.norm_xca(x)))
        x = x.reshape(B, H, W, C)

        x = self.norm(x)
        x = self.mlp(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)
        x = shortcut + self.drop_path(x)
        return x

    SplitTransposeBlock.split_forward = split_forward
    SplitTransposeBlock.forward = split_forward


def install_edgenext_patches():
    install_xca_patch()
    install_sdta_patch()


def _strip_prefix(sd):
    for p in ("module.", "model."):
        if all(k.startswith(p) for k in sd):
            sd = {k[len(p):]: v for k, v in sd.items()}
    return sd


_HEAD_DIM_STAGE1 = {"edgenext_x_small": 16, "edgenext_small": 12, "edgenext_base": 20}


def _infer_arch(sd, dims, depths):
    arch = ARCH_BY_STAGE4_DIM.get(dims[3])
    if arch is not None:
        return arch
    if tuple(depths) == (2, 2, 6, 2):
        return "edgenext_xx_small"
    p = "stages.1.blocks.%d.xca." % (depths[1] - 1)
    if (p + "qkv.weight") in sd:
        hd = int(sd[p + "qkv.weight"].shape[0]) // (3 * int(sd[p + "temperature"].shape[0]))
        for arch, h in _HEAD_DIM_STAGE1.items():
            if h == hd:
                return arch
    raise ValueError("cannot infer the EdgeNeXt variant from this state_dict; pass --arch")


def rebuild_from_state_dict(sd, arch=None):
    """Build a timm EdgeNeXt whose every layer matches the shapes in `sd`
    (pruned stage widths, per-block MLP widths, pruned heads, pruned SDTA
    split widths), then load it strictly."""
    import timm
    import torch
    import torch.nn as nn
    install_edgenext_patches()
    sd = _strip_prefix(dict(sd))
    if "head.fc.weight" not in sd or "stem.0.weight" not in sd:
        raise ValueError("state_dict does not look like a timm EdgeNeXt (missing stem.0 / head.fc)")
    num_classes = int(sd["head.fc.weight"].shape[0])
    dims, depths = [], []
    for s in range(4):
        w = sd["stem.0.weight"] if s == 0 else sd["stages.%d.downsample.1.weight" % s]
        dims.append(int(w.shape[0]))
        b = 0
        while ("stages.%d.blocks.%d.mlp.fc1.weight" % (s, b)) in sd:
            b += 1
        depths.append(b)
    if arch is None:
        arch = _infer_arch(sd, dims, depths)
    model = timm.create_model(arch, pretrained=False, num_classes=num_classes, dims=tuple(dims))
    for s, stage in enumerate(model.stages):
        if len(stage.blocks) != depths[s]:
            raise ValueError("block count mismatch in stage %d (%d vs %d): wrong --arch?"
                             % (s, len(stage.blocks), depths[s]))
        for b, blk in enumerate(stage.blocks):
            p = "stages.%d.blocks.%d." % (s, b)
            w1, w2 = sd[p + "mlp.fc1.weight"], sd[p + "mlp.fc2.weight"]
            blk.mlp.fc1 = nn.Linear(int(w1.shape[1]), int(w1.shape[0]), bias=(p + "mlp.fc1.bias") in sd)
            blk.mlp.fc2 = nn.Linear(int(w2.shape[1]), int(w2.shape[0]), bias=(p + "mlp.fc2.bias") in sd)
            if (p + "xca.qkv.weight") in sd:
                q, pr = sd[p + "xca.qkv.weight"], sd[p + "xca.proj.weight"]
                heads = int(sd[p + "xca.temperature"].shape[0])
                blk.xca.qkv = nn.Linear(int(q.shape[1]), int(q.shape[0]), bias=(p + "xca.qkv.bias") in sd)
                blk.xca.proj = nn.Linear(int(pr.shape[1]), int(pr.shape[0]), bias=(p + "xca.proj.bias") in sd)
                blk.xca.num_heads = heads
                blk.xca.temperature = nn.Parameter(torch.ones(heads, 1, 1))
                convs = []
                for j in range(len(blk.convs)):
                    cw = sd[p + "convs.%d.weight" % j]
                    width = int(cw.shape[0])
                    convs.append(nn.Conv2d(width, width, kernel_size=int(cw.shape[-1]),
                                           padding=int(cw.shape[-1]) // 2, groups=width,
                                           bias=(p + "convs.%d.bias" % j) in sd))
                blk.convs = nn.ModuleList(convs)
                blk.width = int(convs[0].in_channels)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise ValueError("rebuild mismatch: missing=%s unexpected=%s" % (missing[:8], unexpected[:8]))
    return model.eval(), arch


def load_torch_model(path, arch=None):
    """Accepts: a pickled timm EdgeNeXt (your pruning scripts' output), a plain
    state_dict, or this script's own {format, arch, state_dict} checkpoint.
    Always returns a model rebuilt from its state_dict, so every model goes
    through the same code path."""
    import torch
    install_edgenext_patches()
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(obj, torch.nn.Module):
        sd = obj.state_dict()
        arch = arch or None
    elif isinstance(obj, dict) and "state_dict" in obj:
        sd = obj["state_dict"]
        arch = arch or obj.get("arch")
    elif isinstance(obj, dict):
        sd = obj
    else:
        raise ValueError("unrecognised checkpoint type %r in %s" % (type(obj), path))
    model, arch = rebuild_from_state_dict(sd, arch)
    if isinstance(obj, torch.nn.Module):
        obj.eval()
        x = torch.randn(2, 3, 224, 224, generator=torch.Generator().manual_seed(0))
        with torch.inference_mode():
            d = (obj(x) - model(x)).abs().max().item()
        if d > 1e-5:
            raise ValueError("rebuilt model differs from pickled model (max abs diff %g)" % d)
    return model, arch


def count_params(model):
    return int(sum(p.numel() for p in model.parameters()))


def count_macs(model, input_size=224):
    """MACs of one 1x3xHxW forward pass: Conv2d + Linear (all, incl. positional
    encoding projection and head) + the two attention matmuls of every XCA
    (q@k^T and attn@v). Element-wise ops, norms and softmax are not counted.
    FLOPs are reported as 2 x MACs. Same definition for pruned and unpruned."""
    import torch
    import torch.nn as nn
    from timm.models.edgenext import CrossCovarianceAttn
    total = {"conv": 0, "linear": 0, "attn_matmul": 0}
    hooks = []

    def conv_hook(m, inp, out):
        total["conv"] += out.numel() * (m.in_channels // m.groups) * m.kernel_size[0] * m.kernel_size[1]

    def lin_hook(m, inp, out):
        total["linear"] += out.numel() * m.in_features

    def xca_hook(m, inp, out):
        B, N, _ = inp[0].shape
        hd = m.qkv.out_features // 3 // m.num_heads
        total["attn_matmul"] += 2 * B * m.num_heads * hd * hd * N

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            hooks.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(lin_hook))
        elif isinstance(m, CrossCovarianceAttn):
            hooks.append(m.register_forward_hook(xca_hook))
    with torch.inference_mode():
        model(torch.zeros(1, 3, input_size, input_size))
    for h in hooks:
        h.remove()
    macs = sum(total.values())
    return {"macs": int(macs), "flops_2x_macs": int(2 * macs), "breakdown": total,
            "definition": "Conv2d + Linear + XCA attention matmuls, batch 1; FLOPs = 2 x MACs"}


def arch_summary(model):
    """Per stage: trunk width, MLP hidden width per block, heads and the SDTA
    split [conv width x n_convs, pass-through width] of each SDTA block."""
    out = {"stage_dims": [], "mlp_hidden": [], "heads": [], "sdta_split": []}
    for s, stage in enumerate(model.stages):
        out["stage_dims"].append(int(stage.blocks[0].mlp.fc2.out_features))
        out["mlp_hidden"].append([int(b.mlp.fc1.out_features) for b in stage.blocks])
        out["heads"].append([int(b.xca.num_heads) for b in stage.blocks if hasattr(b, "xca")])
        split = []
        for b in stage.blocks:
            if hasattr(b, "convs"):
                w, n = int(b.convs[0].in_channels), len(b.convs)
                split.append([w] * n + [int(b.norm_xca.normalized_shape[0]) - w * n])
        out["sdta_split"].append(split)
    return out


def save_portable_checkpoint(model, arch, path):
    import torch
    torch.save({"format": SCHEMA, "arch": arch, "num_classes": int(model.head.fc.out_features),
                "state_dict": model.state_dict()}, path)



class TorchRunner:
    name = "pytorch"

    def __init__(self, path, threads, arch=None):
        import torch
        torch.set_num_threads(threads)
        self.torch = torch
        self.model, _ = load_torch_model(path, arch)
        self.max_batch = 32
        self.version = torch.__version__
        self.io = {"layout": "NCHW", "input_dtype": "float32"}

    def prepare(self, x):
        return self.torch.from_numpy(np.ascontiguousarray(x))

    def run(self, xt):
        with self.torch.inference_mode():
            return self.model(xt)

    def logits(self, out):
        return out.detach().cpu().numpy().astype(np.float32)


class OrtRunner:
    name = "onnx"

    def __init__(self, path, threads):
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(path, sess_options=so, providers=["CPUExecutionProvider"])
        i = self.sess.get_inputs()[0]
        self.in_name = i.name
        self.out_name = self.sess.get_outputs()[0].name
        self.max_batch = 32 if not isinstance(i.shape[0], int) else 1
        self.version = ort.__version__
        self.io = {"layout": "NCHW", "input_dtype": "float32", "provider": "CPUExecutionProvider",
                   "input_shape": [str(d) for d in i.shape]}

    def prepare(self, x):
        return np.ascontiguousarray(x, np.float32)

    def run(self, x):
        return self.sess.run([self.out_name], {self.in_name: x})[0]

    def logits(self, out):
        return np.asarray(out, np.float32)


def _tflite_interpreter_cls():
    for mod, label in (("ai_edge_litert.interpreter", "ai-edge-litert"),
                       ("tflite_runtime.interpreter", "tflite-runtime"),
                       ("tensorflow.lite", "tensorflow")):
        try:
            m = __import__(mod, fromlist=["Interpreter"])
            return m.Interpreter, label
        except Exception:
            continue
    raise ImportError("no TFLite interpreter (install ai-edge-litert or tflite-runtime)")


class TfliteRunner:
    name = "tflite"

    def __init__(self, path, threads):
        Interpreter, label = _tflite_interpreter_cls()
        self.it = Interpreter(model_path=path, num_threads=threads)
        self.it.allocate_tensors()
        i = self.it.get_input_details()[0]
        o = self.it.get_output_details()[0]
        self.in_idx, self.out_idx = i["index"], o["index"]
        shape = list(i["shape"])
        self.nhwc = len(shape) == 4 and shape[-1] == 3 and shape[1] != 3
        self.in_dtype, self.out_dtype = i["dtype"], o["dtype"]
        self.in_q, self.out_q = i["quantization"], o["quantization"]
        self.max_batch = 1
        self.version = "%s %s" % (label, pkg_version(label) or "")
        self.io = {"layout": "NHWC" if self.nhwc else "NCHW", "input_dtype": np.dtype(self.in_dtype).name,
                   "output_dtype": np.dtype(self.out_dtype).name, "input_quant": list(map(float, self.in_q)),
                   "output_quant": list(map(float, self.out_q)), "input_shape": [int(d) for d in shape]}

    def prepare(self, x):
        if self.nhwc:
            x = x.transpose(0, 2, 3, 1)
        if np.issubdtype(self.in_dtype, np.integer):
            scale, zp = self.in_q
            if not scale:
                raise RuntimeError("quantized input without scale")
            info = np.iinfo(self.in_dtype)
            x = np.clip(np.round(x / scale + zp), info.min, info.max)
        return np.ascontiguousarray(x.astype(self.in_dtype))

    def run(self, x):
        self.it.set_tensor(self.in_idx, x)
        self.it.invoke()
        return self.it.get_tensor(self.out_idx)

    def logits(self, out):
        out = np.asarray(out)
        if np.issubdtype(self.out_dtype, np.integer):
            scale, zp = self.out_q
            return ((out.astype(np.float32) - zp) * scale).astype(np.float32)
        return out.astype(np.float32)


def make_runner(runtime, path, threads, arch=None):
    if runtime == "pytorch":
        return TorchRunner(path, threads, arch)
    if runtime == "onnx":
        return OrtRunner(path, threads)
    if runtime in ("tflite_fp32", "tflite_int8"):
        return TfliteRunner(path, threads)
    raise ValueError(runtime)


def run_logits(runner, x_batch):
    outs = []
    for i in range(0, x_batch.shape[0], runner.max_batch):
        xb = runner.prepare(x_batch[i:i + runner.max_batch])
        outs.append(runner.logits(runner.run(xb)))
    return np.concatenate(outs, 0)


def evaluate_accuracy(runner, images_u8, labels, cfg, chunk=64, progress_every=1000):
    """Returns (top-1 predictions, top-1 correct, top-5 correct) per image."""
    preds = np.empty(labels.size, np.int64)
    top5 = np.zeros(labels.size, bool)
    t0 = time.perf_counter()
    for s in range(0, labels.size, chunk):
        x = preprocess_batch(images_u8[s:s + chunk], cfg)
        lg = run_logits(runner, x)
        n = x.shape[0]
        preds[s:s + n] = lg.argmax(1)
        k = min(5, lg.shape[1])
        top = np.argpartition(-lg, k - 1, axis=1)[:, :k]
        top5[s:s + n] = (top == labels[s:s + n, None]).any(1)
        done = s + n
        if progress_every and (done // progress_every) != ((done - n) // progress_every):
            log("      %d/%d images (%.0f s)" % (done, labels.size, time.perf_counter() - t0))
    return preds, preds == labels, top5


def measure_sustained(runners, x_single, seconds, meter, idle_seconds):
    """Continuous batch-1 inference for `seconds` per model, in the order
    A B B A (half the time each) so heating affects both equally, after an idle
    window. Every window records unix start/end times (to match an external
    power meter's log), inference count, temperatures and, if the device has an
    on-board sensor, energy."""
    prepared = {k: r.prepare(x_single) for k, r in runners.items()}
    st = device_state()
    tok = meter.start()
    t0 = time.time()
    time.sleep(idle_seconds)
    idle = {"start_unix": t0, "end_unix": time.time(), "seconds": idle_seconds, "energy": meter.stop(tok),
            "temp_start_c": st["max_temp_c"], "temp_end_c": device_state()["max_temp_c"]}
    keys = list(runners)
    windows = []
    for k in keys + keys[::-1]:
        r, xp = runners[k], prepared[k]
        st0 = device_state()
        tok = meter.start()
        t_unix, tp, n = time.time(), time.perf_counter(), 0
        while time.perf_counter() - tp < seconds / 2.0:
            r.run(xp)
            n += 1
        el = time.perf_counter() - tp
        e = meter.stop(tok)
        st1 = device_state()
        windows.append({"model": k, "start_unix": t_unix, "end_unix": time.time(), "seconds": el, "inferences": n,
                        "energy": e, "temp_start_c": st0["max_temp_c"], "temp_end_c": st1["max_temp_c"],
                        "cpu_mhz_end": st1["cpu_mhz"], "throttled_end": st1["throttled"]})
    per = {}
    idle_w = idle["energy"]["avg_w"] if idle["energy"] else None
    for k in keys:
        ws = [w for w in windows if w["model"] == k]
        n, sec = sum(w["inferences"] for w in ws), sum(w["seconds"] for w in ws)
        d = {"inferences": n, "seconds": sec, "throughput_img_s": n / sec if sec else None,
             "mean_latency_ms": 1000.0 * sec / n if n else None}
        if all(w["energy"] for w in ws):
            j = sum(w["energy"]["joules"] for w in ws)
            d.update(energy_j=j, avg_power_w=j / sec, energy_mj_per_inference=1000.0 * j / n)
            if idle_w is not None:
                d["energy_mj_per_inference_above_idle"] = 1000.0 * (j - idle_w * sec) / n
        per[k] = d
    return {"idle": idle, "windows": windows, "per_model": per, "meter": meter.kind,
            "meter_detail": meter.detail}


def run_probe(runtime, path, threads, arch, x_path, steady=10, timeout=900):
    """Load and run one model in a fresh process: model load time, first
    (cold-start) inference time and memory, without other models in memory."""
    cmd = [sys.executable, os.path.abspath(__file__), "probe", "--runtime", runtime, "--model-file", path,
           "--threads", str(threads), "--input", x_path, "--steady", str(steady)]
    if arch:
        cmd += ["--arch", arch]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    for line in r.stdout.splitlines():
        if line.startswith("PROBE_JSON "):
            return json.loads(line[len("PROBE_JSON "):])
    raise RuntimeError("probe failed: %s" % (r.stderr.strip().splitlines()[-1:] or ["no output"])[0])


def cmd_probe(a):
    rss0 = rss_bytes()
    t0 = time.perf_counter()
    if a.runtime == "pytorch":
        import torch
        import timm
    elif a.runtime == "onnx":
        import onnxruntime
    else:
        _tflite_interpreter_cls()
    import_ms = 1000 * (time.perf_counter() - t0)
    rss_import = rss_bytes()
    x = np.load(a.input)
    t0 = time.perf_counter()
    r = make_runner(a.runtime, a.model_file, a.threads, a.arch)
    load_ms = 1000 * (time.perf_counter() - t0)
    rss_load = rss_bytes()
    xp = r.prepare(x)
    t0 = time.perf_counter()
    r.run(xp)
    first_ms = 1000 * (time.perf_counter() - t0)
    steady = []
    for _ in range(a.steady):
        t0 = time.perf_counter()
        r.run(xp)
        steady.append(1000 * (time.perf_counter() - t0))
    out = {"rss_start_bytes": rss0, "runtime_import_ms": import_ms, "rss_after_import_bytes": rss_import,
           "load_ms": load_ms, "rss_after_load_bytes": rss_load, "first_inference_ms": first_ms,
           "steady_median_ms": float(np.median(steady)) if steady else None,
           "rss_end_bytes": rss_bytes(), "peak_rss_bytes": peak_rss_bytes()}
    if out["peak_rss_bytes"] is not None and rss_import is not None:
        out["peak_above_runtime_bytes"] = out["peak_rss_bytes"] - rss_import
    print("PROBE_JSON " + json.dumps(out), flush=True)


def measure_latency_pair(runners, x_single, warmup, runs, rounds):
    """Batch-1 latency for several models of ONE runtime, interleaved in rounds
    (order rotated each round) so thermal throttling and background load hit all
    models equally. Times only the inference call (set_tensor+invoke+get_tensor
    for TFLite, session.run for ONNX Runtime, model(x) for PyTorch)."""
    prepared = {k: r.prepare(x_single) for k, r in runners.items()}
    for k, r in runners.items():
        for _ in range(warmup):
            r.run(prepared[k])
    lat = {k: [] for k in runners}
    per_round = max(1, runs // rounds)
    keys = list(runners)
    for rd in range(rounds):
        order = keys[rd % len(keys):] + keys[:rd % len(keys)]
        for k in order:
            r, xp = runners[k], prepared[k]
            for _ in range(per_round):
                t0 = time.perf_counter_ns()
                r.run(xp)
                lat[k].append((time.perf_counter_ns() - t0) / 1e6)
    return lat



def export_onnx(model, path, input_size, opset):
    import torch
    x = torch.zeros(1, 3, input_size, input_size)
    kwargs = dict(export_params=True, opset_version=opset, do_constant_folding=True,
                  input_names=["input"], output_names=["output"],
                  dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}})
    try:
        torch.onnx.export(model, x, path, dynamo=False, **kwargs)
    except TypeError:
        torch.onnx.export(model, x, path, **kwargs)
    import onnx
    m = onnx.load(path)
    onnx.checker.check_model(m)
    if any(t.data_location == onnx.TensorProto.EXTERNAL for t in m.graph.initializer):
        raise RuntimeError("ONNX export wrote external data; keep models self-contained")


def export_tflite_fp32(model, path, input_size):
    import torch
    try:
        import litert_torch as conv
    except ImportError:
        import ai_edge_torch as conv
    sample = (torch.zeros(1, 3, input_size, input_size),)
    edge = conv.convert(model.eval(), sample)
    edge.export(path)


def _rank_fix(mode):
    """Replacement for ai_edge_quantizer's fix_quantization_params_rank.

    The quantizer sometimes reports quantized_dimension=0 for per-channel
    constants shaped [1, 1, C] / [1, 1, 1, C] (EdgeNeXt's LayerScale gamma).
    mode="legacy" is the shim from your step2_convert_int8.py: it puts the C
    scales on axis 0, the [1,..,C] tensor broadcasts against [C,..,1] and the
    stored constant becomes C x C values, of which TFLite reads only the first
    row (wrong for every channel but one). mode="axis_match" (default) puts the
    scales on the axis whose size equals the number of scales."""
    import dataclasses

    def fix(tensor_data, qp):
        scales, zps = np.asarray(qp.scale), np.asarray(qp.zero_point)
        if mode == "legacy":
            if tensor_data.ndim == scales.ndim:
                return qp
            if tensor_data.ndim == 0:
                scales, zps = np.array(scales.item()), np.array(zps.item())
            else:
                scales, zps = np.squeeze(scales), np.squeeze(zps)
                if scales.ndim == 0:
                    scales = np.reshape(scales, [1] * tensor_data.ndim)
                    zps = np.reshape(zps, [1] * tensor_data.ndim)
                else:
                    shape = [1] * tensor_data.ndim
                    shape[qp.quantized_dimension or 0] = scales.shape[0]
                    scales, zps = np.reshape(scales, shape), np.reshape(zps, shape)
            return dataclasses.replace(qp, scale=scales, zero_point=zps)
        if tensor_data.ndim == 0:
            if scales.size != 1:
                raise ValueError("scalar tensor with %d scales" % scales.size)
            return dataclasses.replace(qp, scale=np.array(scales.item()), zero_point=np.array(zps.item()))
        n = int(scales.size)
        if zps.size == 1 and n > 1:
            zps = np.full(n, zps.reshape(-1)[0], dtype=zps.dtype)
        shape = [1] * tensor_data.ndim
        qd = qp.quantized_dimension
        if n > 1:
            if qd is None or qd >= tensor_data.ndim or tensor_data.shape[qd] != n:
                match = [d for d, s in enumerate(tensor_data.shape) if s == n]
                if not match:
                    raise ValueError("cannot place %d scales on a tensor of shape %s" % (n, tensor_data.shape))
                qd = match[-1]
            shape[qd] = n
        return dataclasses.replace(qp, scale=scales.reshape(shape), zero_point=zps.reshape(shape),
                                   quantized_dimension=qd)
    return fix


def tflite_malformed_constants(path):
    """Count constant tensors whose stored bytes do not match their shape x dtype
    (the symptom of the legacy rank fix). Returns None if flatbuffer tools are missing."""
    try:
        from ai_edge_litert.tools import flatbuffer_utils as fbu
    except ImportError:
        return None
    itemsize = {0: 4, 1: 2, 2: 4, 3: 1, 4: 8, 7: 2, 9: 1}
    m = fbu.read_model(path)
    bad = 0
    for sg in m.subgraphs:
        for t in sg.tensors:
            buf = m.buffers[t.buffer].data if t.buffer < len(m.buffers) else None
            if buf is None or len(buf) == 0 or t.type not in itemsize:
                continue
            if len(buf) != itemsize[t.type] * int(np.prod(t.shape if len(t.shape) else [1])):
                bad += 1
    return bad


def quantize_tflite_int8(fp32_path, out_path, calib_x, rank_fix="axis_match"):
    """Full static INT8 with AI Edge Quantizer, positional encoding kept FP32,
    same recipe as your step2_convert_int8.py (with the corrected rank fix)."""
    from ai_edge_quantizer import quantizer, recipe_manager, qtyping, recipe
    try:
        from ai_edge_quantizer.algorithms.uniform_quantize import uniform_quantize_tensor as uqt
        if not hasattr(uqt, "_edgenext_bench_original"):
            uqt._edgenext_bench_original = uqt.fix_quantization_params_rank
        uqt.fix_quantization_params_rank = _rank_fix(rank_fix) if rank_fix else uqt._edgenext_bench_original
    except ImportError:
        pass

    Interpreter, _ = _tflite_interpreter_cls()
    sigs = Interpreter(model_path=fp32_path).get_signature_list()
    in_name = "args_0"
    if sigs:
        first = sigs[next(iter(sigs))]
        if first.get("inputs"):
            in_name = first["inputs"][0]

    rp = recipe_manager.RecipeManager()
    rp.add_static_config(regex=".*", operation_name=qtyping.TFLOperationName.ALL_SUPPORTED,
                         activation_num_bits=8, weight_num_bits=8)
    rp.add_quantization_config(regex=".*PositionalEncodingFourier.*",
                               operation_name=qtyping.TFLOperationName.ALL_SUPPORTED,
                               op_config=None, algorithm_key=recipe.AlgorithmName.NO_QUANTIZE)
    q = quantizer.Quantizer(fp32_path)
    q.load_quantization_recipe(rp.get_quantization_recipe())
    calib = {None: [{in_name: calib_x[i:i + 1]} for i in range(calib_x.shape[0])]}
    res = q.quantize(q.calibrate(calib))
    tmp = out_path + ".tmp.tflite"
    res.export_model(tmp, overwrite=True)
    try:
        from ai_edge_litert.tools import flatbuffer_utils as fbu
        m = fbu.read_model(tmp)
        for t in m.subgraphs[0].tensors:
            qn = t.quantization
            if qn is not None and qn.scale is not None and len(qn.scale) > 1:
                if t.shape[qn.quantizedDimension] != len(qn.scale):
                    match = [d for d, s in enumerate(t.shape) if s == len(qn.scale)]
                    if match:
                        qn.quantizedDimension = match[0]
        fbu.write_model(m, out_path)
        os.remove(tmp)
    except ImportError:
        os.replace(tmp, out_path)



def cmd_export(a):
    os.environ.setdefault("TQDM_DISABLE", "1")
    import torch
    torch.manual_seed(a.seed)
    out = os.path.abspath(a.out)
    os.makedirs(os.path.join(out, "models"), exist_ok=True)
    ds = DATASETS[a.dataset]
    cfg = dict(ds["preprocess"])
    cfg["size"] = a.input_size
    class_order = a.class_order.split(",") if a.class_order else None
    warnings = []

    log("[1/7] loading models")
    models = {}
    for role, path in (("pruned", a.pruned), ("unpruned", a.unpruned)):
        m, arch = load_torch_model(path, a.arch)
        models[role] = {"model": m, "arch": arch, "source": os.path.abspath(path),
                        "source_sha256": sha256_file(path)}
        log("  %s: %s (%s)" % (role, os.path.basename(path), arch))
    if models["pruned"]["arch"] != models["unpruned"]["arch"]:
        if not a.allow_arch_mismatch:
            raise SystemExit("pruned and unpruned are different architectures: %s vs %s (pass "
                             "--allow-arch-mismatch to compare a pruned model with a smaller variant)"
                             % (models["pruned"]["arch"], models["unpruned"]["arch"]))
        warnings.append("reference ('unpruned') model is %s, pruned model is %s"
                        % (models["unpruned"]["arch"], models["pruned"]["arch"]))
    if models["pruned"]["source_sha256"] == models["unpruned"]["source_sha256"]:
        raise SystemExit("pruned and unpruned are the same file")
    n_classes = dataset_num_classes(a.dataset, a.data_root)
    for role in models:
        nc = int(models[role]["model"].head.fc.out_features)
        if nc != n_classes:
            raise SystemExit("%s model has %d outputs but %s has %d classes"
                             % (role, nc, a.dataset, n_classes))

    log("[2/7] parameters and MACs")
    for role, d in models.items():
        d["params"] = count_params(d["model"])
        d["complexity"] = count_macs(d["model"], a.input_size)
        d["arch_summary"] = arch_summary(d["model"])
        log("  %s: %s params, %.2f M MACs" % (role, f"{d['params']:,}", d["complexity"]["macs"] / 1e6))
    ratio = models["pruned"]["params"] / models["unpruned"]["params"]
    if ratio > 0.99:
        warnings.append("pruned model keeps %.2f%% of the unpruned parameters - it is essentially "
                        "unpruned" % (100 * ratio))

    log("[3/7] evaluation and calibration images")
    eval_imgs, eval_labels, class_names = load_raw_split(a.dataset, a.data_root, ds["eval_split"],
                                                         class_order, a.download, per_class=a.eval_per_class)
    if a.dataset in ("tiny10", "tiny200"):
        class_order = class_names
    if a.dataset in ("cifar10", "cifar100"):
        train_imgs, train_labels, _ = load_raw_split(a.dataset, a.data_root, "train", class_order, a.download)
        calib_idx = stratified_indices(train_labels, total=a.calib_n, num_classes=n_classes, spread=True)
        calib_rule = "class-balanced, evenly spaced within each class"
    else:
        k = max(1, int(math.ceil(a.calib_n / float(n_classes))))
        train_imgs, train_labels, _ = load_raw_split(a.dataset, a.data_root, "train", class_order,
                                                     a.download, per_class=k)
        calib_idx = np.arange(train_labels.size)
        calib_rule = "first %d images (sorted file names) of every class; indices refer to that list" % k
    calib_x = preprocess_batch(train_imgs[calib_idx], cfg)
    ref_x = preprocess_batch(eval_imgs[:4], cfg)
    evalset = os.path.join(out, "evalset.npz")
    np.savez_compressed(evalset, images=eval_imgs, labels=eval_labels, ref_preprocessed=ref_x)
    log("  eval: %d %s images; calibration: %d train images" % (eval_labels.size, ds["eval_split"], calib_idx.size))

    check_x = preprocess_batch(eval_imgs[:16], cfg)
    xt = torch.from_numpy(check_x)
    for role, d in models.items():
        with torch.inference_mode():
            d["ref_logits"] = d["model"](xt).numpy()

    log("[4/7] PyTorch checkpoints")
    for role, d in models.items():
        p = os.path.join(out, "models", "%s.pt" % role)
        save_portable_checkpoint(d["model"], d["arch"], p)
        d["files"] = {"pytorch": file_record(p, out)}

    log("[5/7] ONNX")
    for role, d in models.items():
        p = os.path.join(out, "models", "%s.onnx" % role)
        export_onnx(d["model"], p, a.input_size, a.opset)
        d["files"]["onnx"] = file_record(p, out)
        r = OrtRunner(p, threads=1)
        d.setdefault("export_checks", {})["onnx_max_abs_diff_vs_torch"] = float(
            np.abs(run_logits(r, check_x) - d["ref_logits"]).max())

    if not a.skip_tflite:
        log("[6/7] TFLite FP32")
        for role, d in models.items():
            p = os.path.join(out, "models", "%s_fp32.tflite" % role)
            export_tflite_fp32(d["model"], p, a.input_size)
            d["files"]["tflite_fp32"] = file_record(p, out)
            r = TfliteRunner(p, threads=1)
            d["export_checks"]["tflite_fp32_max_abs_diff_vs_torch"] = float(
                np.abs(run_logits(r, check_x) - d["ref_logits"]).max())
            d.setdefault("tflite_memory", {})["fp32"] = tflite_memory_estimate(p)
        if not a.skip_int8:
            log("[7/7] TFLite INT8 (full integer, %d calibration images)" % calib_idx.size)
            for role, d in models.items():
                fp32 = os.path.join(out, d["files"]["tflite_fp32"]["file"])
                p = os.path.join(out, "models", "%s_int8.tflite" % role)
                quantize_tflite_int8(fp32, p, calib_x)
                d["files"]["tflite_int8"] = file_record(p, out)
                r = TfliteRunner(p, threads=1)
                agree = run_logits(r, check_x).argmax(1) == d["ref_logits"].argmax(1)
                d["export_checks"]["tflite_int8_top1_agreement_vs_torch_on_16"] = float(agree.mean())
                bad = tflite_malformed_constants(p)
                d["export_checks"]["tflite_int8_malformed_constants"] = bad
                d.setdefault("tflite_memory", {})["int8"] = tflite_memory_estimate(p)
                if bad:
                    warnings.append("%s INT8 model has %d malformed constant tensors" % (role, bad))

    for role, d in models.items():
        for k, v in d.get("export_checks", {}).items():
            if "max_abs_diff" in k and v > 1e-3:
                warnings.append("%s %s = %.2e (export not numerically faithful)" % (role, k, v))

    script_copy = os.path.join(out, "edgenext_bench.py")
    if os.path.abspath(__file__) != script_copy:
        shutil.copy2(os.path.abspath(__file__), script_copy)

    manifest = {
        "schema": SCHEMA, "created": now_iso(), "experiment": a.experiment,
        "host_env": env_info(),
        "script_sha256": sha256_file(script_copy),
        "dataset": {"name": a.dataset, "eval_split": ds["eval_split"], "num_classes": n_classes,
                    "class_names": class_names, "n_eval": int(eval_labels.size),
                    "eval_per_class_cap": a.eval_per_class,
                    "preprocess": cfg, "evalset": file_record(evalset, out)},
        "calibration": {"split": "train", "n": int(calib_idx.size), "indices": calib_idx.tolist(),
                        "rule": calib_rule},
        "models": {role: {k: v for k, v in d.items() if k not in ("model", "ref_logits")}
                   for role, d in models.items()},
        "warnings": warnings,
    }
    write_json(os.path.join(out, "manifest.json"), manifest)
    log("\nwrote %s" % os.path.join(out, "manifest.json"))
    for w in warnings:
        log("WARNING: " + w)



def cmd_bench(a):
    import gc
    import tempfile
    art = os.path.abspath(a.artifacts)
    man_path = os.path.join(art, "manifest.json")
    man = read_json(man_path)
    if man.get("schema") != SCHEMA:
        raise SystemExit("unknown manifest schema")
    threads = a.threads or min(4, os.cpu_count() or 1)

    log("verifying checksums")
    ev = man["dataset"]["evalset"]
    if sha256_file(os.path.join(art, ev["file"])) != ev["sha256"]:
        raise SystemExit("evalset.npz does not match the manifest")
    for role, m in man["models"].items():
        for rt, rec in m["files"].items():
            p = os.path.join(art, rec["file"])
            if os.path.exists(p) and sha256_file(p) != rec["sha256"]:
                raise SystemExit("%s %s file %s does not match the manifest" % (role, rt, rec["file"]))

    data = np.load(os.path.join(art, ev["file"]))
    images, labels = data["images"], data["labels"]
    cfg = man["dataset"]["preprocess"]
    nc = man["dataset"]["num_classes"]
    idx = stratified_indices(labels, per_class=a.per_class, total=a.limit, num_classes=nc)
    subset = idx.size < labels.size
    images, labels = images[idx], labels[idx]
    pre_diff = float(np.abs(preprocess_batch(data["images"][:4], cfg) - data["ref_preprocessed"]).max())
    notes = []
    if sha256_file(os.path.abspath(__file__)) != man.get("script_sha256"):
        notes.append("this script differs from the copy saved at export (use the one inside the "
                     "artifacts folder)")
    if pre_diff > 1e-4:
        notes.append("preprocessing on this device differs from the export host (max abs %.2e); "
                     "check the Pillow version" % pre_diff)
    log("evaluating on %d of %d %s images%s" % (labels.size, data["labels"].size,
                                                 man["dataset"]["eval_split"], " (SUBSET)" if subset else ""))

    hints = detect_emulation()
    emulated = {"yes": True, "no": False}.get(a.emulated, looks_emulated(hints))
    wanted = a.runtimes.split(",") if a.runtimes else ALL_RUNTIMES
    x_single = preprocess(images[0], cfg)[None]
    meter = PowerMeter()
    if a.sustained_seconds > 0:
        log("power sensor: %s" % (meter.detail or meter.reason))
    state_start = device_state()
    runs, runtime_info = [], []
    tmpdir = tempfile.mkdtemp(prefix="bench_probe_")
    x_path = os.path.join(tmpdir, "x.npy")
    np.save(x_path, x_single)

    for rt in wanted:
        log("\n== %s ==" % rt)
        info = {"runtime": rt, "state_before": device_state()}
        runners, recs = {}, {}
        for role, m in man["models"].items():
            rec = {"runtime": rt, "model": role}
            f = m["files"].get(rt)
            if f is None or not os.path.exists(os.path.join(art, f["file"])):
                rec.update(status="skipped", reason="no %s artifact in this package" % rt)
            else:
                rec.update(file=f["file"], file_sha256=f["sha256"], file_bytes=f["bytes"])
                try:
                    runners[role] = make_runner(rt, os.path.join(art, f["file"]), threads, m.get("arch"))
                    rec.update(status="ok", runtime_version=runners[role].version, io=runners[role].io)
                except Exception as e:
                    rec.update(status="skipped", reason="%s: %s" % (type(e).__name__, e))
            recs[role] = rec
        try:
            for role, r in runners.items():
                out_dim = run_logits(r, x_single).shape[1]
                if out_dim != nc:
                    raise RuntimeError("%s outputs %d classes, manifest says %d" % (role, out_dim, nc))
                log("  accuracy: %s" % role)
                preds, correct, c5 = evaluate_accuracy(r, images, labels, cfg, progress_every=a.progress)
                k, k5, n = int(correct.sum()), int(c5.sum()), int(correct.size)
                recs[role]["accuracy"] = {"correct": k, "n": n, "top1": k / n, "wilson95": wilson_interval(k, n),
                                          "top5_correct": k5, "top5": k5 / n, "top5_wilson95": wilson_interval(k5, n)}
                recs[role]["predictions"] = preds.tolist()
                log("    top-1 %d/%d = %.2f%%   top-5 %.2f%%" % (k, n, 100.0 * k / n, 100.0 * k5 / n))
            if runners:
                log("  latency: %d warm-up + %d timed runs per model, %d interleaved rounds, %d threads"
                    % (a.warmup, a.runs, a.rounds, threads))
                lat = measure_latency_pair(runners, x_single, a.warmup, a.runs, a.rounds)
                for role, ms in lat.items():
                    recs[role]["latency_ms"] = latency_summary(ms)
                    recs[role]["latency_raw_ms"] = [round(v, 4) for v in ms]
                    log("    %s: median %.3f ms, mean %.3f ms" % (role, np.median(ms), np.mean(ms)))
            if runners and a.sustained_seconds > 0:
                log("  sustained: %.0f s per model (A B B A) after %.0f s idle" % (a.sustained_seconds,
                                                                                  a.idle_seconds))
                sus = measure_sustained(runners, x_single, a.sustained_seconds, meter, a.idle_seconds)
                info["sustained"] = {k: v for k, v in sus.items() if k != "per_model"}
                for role, d in sus["per_model"].items():
                    recs[role]["sustained"] = d
                    log("    %s: %.2f img/s%s" % (role, d["throughput_img_s"] or 0,
                                                 ", %.2f mJ/inference" % d["energy_mj_per_inference"]
                                                 if "energy_mj_per_inference" in d else ""))
        except Exception as e:
            for role in runners:
                recs[role].update(status="error", reason="%s: %s" % (type(e).__name__, e))
                recs[role].pop("accuracy", None)
                recs[role].pop("predictions", None)
            log("  ERROR: %s: %s" % (type(e).__name__, e))
        ok_roles = [r for r in runners if recs[r]["status"] == "ok"]
        del runners
        gc.collect()
        info["state_after"] = device_state()
        if a.mem_probes > 0:
            for role in ok_roles:
                m = man["models"][role]
                pr = []
                for _ in range(a.mem_probes):
                    try:
                        pr.append(run_probe(rt, os.path.join(art, m["files"][rt]["file"]), threads,
                                            m.get("arch"), x_path))
                    except Exception as e:
                        recs[role]["probe_error"] = "%s: %s" % (type(e).__name__, e)
                        break
                if pr:
                    med = {k: float(np.median([p[k] for p in pr if p.get(k) is not None]))
                           for k in pr[0] if any(p.get(k) is not None for p in pr)}
                    recs[role]["probe"] = {"runs": pr, "median": med}
                    log("  %s: load %.1f ms, first inference %.1f ms, peak RSS %.1f MiB (%.1f MiB above the "
                        "runtime)" % (role, med.get("load_ms", float("nan")),
                                      med.get("first_inference_ms", float("nan")),
                                      med.get("peak_rss_bytes", float("nan")) / 2 ** 20,
                                      med.get("peak_above_runtime_bytes", float("nan")) / 2 ** 20))
        for rec in recs.values():
            if rec["status"] != "ok":
                log("  %s: skipped (%s)" % (rec["model"], rec["reason"]))
        runs.extend(recs.values())
        runtime_info.append(info)
    shutil.rmtree(tmpdir, ignore_errors=True)

    result = {
        "schema": SCHEMA, "created": now_iso(), "device_label": a.device_label,
        "emulated": emulated, "emulation_hints": hints, "env": env_info(),
        "experiment": man["experiment"], "manifest_sha256": sha256_file(man_path),
        "script_sha256": sha256_file(os.path.abspath(__file__)),
        "models": {r: {"arch": m["arch"], "params": m["params"], "complexity": m["complexity"],
                       "files": m["files"], "source_sha256": m.get("source_sha256"),
                       "tflite_memory": m.get("tflite_memory")} for r, m in man["models"].items()},
        "num_classes": nc,
        "settings": {"threads": threads, "warmup": a.warmup, "runs": a.runs, "rounds": a.rounds,
                     "batch_size_latency": 1, "timed_section": "inference call only",
                     "sustained_seconds": a.sustained_seconds, "idle_seconds": a.idle_seconds,
                     "mem_probes": a.mem_probes},
        "power_sensor": {"kind": meter.kind, "detail": meter.detail, "reason_if_none": meter.reason},
        "device_state": {"start": state_start, "end": device_state()},
        "runtime_info": runtime_info,
        "eval": {"dataset": man["dataset"]["name"], "split": man["dataset"]["eval_split"],
                 "n_available": int(data["labels"].size), "n_used": int(labels.size), "subset": subset,
                 "indices": idx.tolist(), "labels": labels.tolist(),
                 "preprocess_max_abs_diff_vs_host": pre_diff},
        "notes": notes, "runs": runs,
    }
    out = a.out or os.path.join(art, "results", "%s__%s.json" % (
        a.device_label.replace(" ", "_"), datetime.now().strftime("%Y%m%d-%H%M%S")))
    write_json(out, result)
    log("\nwrote %s" % out)



def _fmt(v, nd=2):
    return "-" if v is None else ("%.*f" % (nd, v))


def _mib(b):
    return b / (1024.0 * 1024.0)


def read_power_csv(path):
    """External power-meter log -> (unix_times, watts). Columns (header, any
    case): a time column (unix_time | timestamp | time | t; unix seconds or ISO
    date-time in this computer's local time) and either watts (watts | power_w |
    power | w) or volts + amps (volts | voltage | v, amps | current | a)."""
    rows = list(csv.DictReader(open(path, newline="", encoding="utf-8-sig")))
    if not rows:
        raise ValueError("empty power log %s" % path)
    cols = {c.strip().lower(): c for c in rows[0]}

    def pick(names):
        return next((cols[n] for n in names if n in cols), None)
    tc = pick(["unix_time", "timestamp", "time", "t"])
    wc = pick(["watts", "power_w", "power", "w"])
    vc, ac = pick(["volts", "voltage", "v"]), pick(["amps", "current", "a"])
    if tc is None or (wc is None and (vc is None or ac is None)):
        raise ValueError("%s needs a time column and watts (or volts + amps)" % path)
    ts, ws = [], []
    for r in rows:
        t = r[tc].strip()
        try:
            tv = float(t)
        except ValueError:
            tv = datetime.fromisoformat(t).timestamp()
        ts.append(tv)
        ws.append(float(r[wc]) if wc else float(r[vc]) * float(r[ac]))
    o = np.argsort(ts)
    return np.asarray(ts)[o], np.asarray(ws)[o]


def _meter_window(ts, ws, start, end):
    sel = (ts >= start) & (ts <= end)
    if sel.sum() < 2:
        return None
    w = float(ws[sel].mean())
    return {"avg_w": w, "joules": w * (end - start), "samples": int(sel.sum())}


def sustained_energy(r, rt, power_logs):
    """Per-model sustained metrics of one runtime in one results file, with
    energy from the on-board sensor or, if given, the external meter log."""
    info = next((x for x in r.get("runtime_info", []) if x["runtime"] == rt and x.get("sustained")), None)
    if info is None:
        return None
    sus = info["sustained"]
    log_ = power_logs.get(r["device_label"])
    src = "external meter" if log_ is not None else (sus.get("meter") or None)
    idle = sus["idle"]
    idle_e = _meter_window(log_[0], log_[1], idle["start_unix"], idle["end_unix"]) if log_ is not None \
        else idle.get("energy")
    out = {"source": src, "idle_w": idle_e["avg_w"] if idle_e else None, "models": {}}
    temps = [w.get("temp_start_c") for w in sus["windows"]] + [w.get("temp_end_c") for w in sus["windows"]]
    temps = [t for t in temps if t is not None]
    out["temp_min_c"], out["temp_max_c"] = (min(temps), max(temps)) if temps else (None, None)
    for role in {w["model"] for w in sus["windows"]}:
        ws = [w for w in sus["windows"] if w["model"] == role]
        n, sec = sum(w["inferences"] for w in ws), sum(w["seconds"] for w in ws)
        es = [(_meter_window(log_[0], log_[1], w["start_unix"], w["end_unix"]) if log_ is not None
               else w.get("energy")) for w in ws]
        d = {"throughput_img_s": n / sec if sec else None, "inferences": n, "seconds": sec}
        if all(es) and n:
            j = sum(e["joules"] for e in es)
            d.update(avg_power_w=j / sec, mj_per_inference=1000.0 * j / n)
            if out["idle_w"] is not None:
                d["mj_per_inference_above_idle"] = 1000.0 * (j - out["idle_w"] * sec) / n
        out["models"][role] = d
    return out


def _pair(u, p, fmt):
    return "%s / %s" % (fmt % u if u is not None else "-", fmt % p if p is not None else "-")


def cmd_report(a):
    files = []
    for pat in a.results:
        files += glob.glob(pat)
    files = sorted(set(files))
    if not files:
        raise SystemExit("no results files")
    results = [read_json(f) for f in files]
    for r, f in zip(results, files):
        r["_file"] = os.path.basename(f)
    exps = sorted({(r["experiment"], r["manifest_sha256"]) for r in results})
    os.makedirs(a.out, exist_ok=True)
    md, rows_csv, warn = [], [], []
    power_logs = {}
    for spec in a.power_csv or []:
        lbl, path = spec.split("=", 1)
        power_logs[lbl] = read_power_csv(path)

    for exp, msha in exps:
        rs = [r for r in results if r["manifest_sha256"] == msha]
        m0 = rs[0]["models"]
        md.append("# %s\n" % exp)
        md.append("Manifest SHA-256 `%s`. All values below are computed from the results files "
                  "listed at the end; none are entered by hand.\n" % msha[:16])

        md.append("## Model complexity (computed at export)\n")
        md.append("| | Unpruned | Pruned | Reduction |")
        md.append("|---|---|---|---|")
        pu, pp = m0["unpruned"]["params"], m0["pruned"]["params"]
        mu, mp = m0["unpruned"]["complexity"]["macs"], m0["pruned"]["complexity"]["macs"]
        md.append("| Parameters | %s | %s | %.2f%% |" % (f"{pu:,}", f"{pp:,}", 100 * (1 - pp / pu)))
        md.append("| MACs (M) | %.2f | %.2f | %.2f%% |" % (mu / 1e6, mp / 1e6, 100 * (1 - mp / mu)))
        for rt in ALL_RUNTIMES:
            fu, fp = m0["unpruned"]["files"].get(rt), m0["pruned"]["files"].get(rt)
            if fu and fp:
                md.append("| %s file (bytes / MiB) | %s / %.2f | %s / %.2f | %.2f%% |" % (
                    rt, f"{fu['bytes']:,}", _mib(fu["bytes"]), f"{fp['bytes']:,}", _mib(fp["bytes"]),
                    100 * (1 - fp["bytes"] / fu["bytes"])))
        for kind in ("fp32", "int8"):
            tu = (m0["unpruned"].get("tflite_memory") or {}).get(kind)
            tp = (m0["pruned"].get("tflite_memory") or {}).get(kind)
            if tu and tp:
                for key, label in (("weights_bytes", "weights"), ("peak_activation_bytes", "peak activation memory")):
                    md.append("| TFLite %s %s (KiB, estimate) | %.1f | %.1f | %.2f%% |" % (
                        kind.upper(), label, tu[key] / 1024, tp[key] / 1024, 100 * (1 - tp[key] / tu[key])))
        md.append("\nMACs: %s. TFLite memory: %s; the microcontroller arena is measured by its firmware.\n"
                  % (m0["unpruned"]["complexity"]["definition"],
                     "liveness estimate of activation tensors over the op order"))
        if pp / pu > 0.99:
            warn.append("%s: pruned model keeps %.2f%% of parameters (essentially unpruned)" % (exp, 100 * pp / pu))

        md.append("## Pruned vs unpruned on each device and runtime\n")
        md.append("| Device | Runtime | Images | Unpruned top-1 | Pruned top-1 | Δ (pp) | McNemar p | "
                  "Unpruned median ms | Pruned median ms | Speedup (95% CI) | Threads |")
        md.append("|---|---|---|---|---|---|---|---|---|---|---|")
        fig_rows = []
        for r in rs:
            dev = r["device_label"] + (" (emulated)" if r["emulated"] else "")
            if r["emulated"]:
                warn.append("%s: %s is emulated - latency is not hardware performance" % (exp, r["device_label"]))
            if r["eval"]["subset"]:
                warn.append("%s: %s used %d of %d images" % (exp, r["device_label"], r["eval"]["n_used"],
                                                             r["eval"]["n_available"]))
            for n in r.get("notes", []):
                warn.append("%s: %s: %s" % (exp, r["device_label"], n))
            states = [r.get("device_state", {}).get("start"), r.get("device_state", {}).get("end")]
            states += [x.get(k) for x in r.get("runtime_info", []) for k in ("state_before", "state_after")]
            states = [x for x in states if x]
            hot = [x["max_temp_c"] for x in states if x.get("max_temp_c") is not None]
            if hot and max(hot) >= 80:
                warn.append("%s: %s reached %.1f °C - check cooling" % (exp, r["device_label"], max(hot)))
            flags = sorted({k for x in states if x.get("throttled") for k, v in x["throttled"].items()
                            if k.endswith("_now") and v})
            if flags:
                warn.append("%s: %s reported %s during the run - latency may be throttled"
                            % (exp, r["device_label"], ", ".join(flags)))
            labels = np.asarray(r["eval"]["labels"])
            for rt in ALL_RUNTIMES:
                by = {x["model"]: x for x in r["runs"] if x["runtime"] == rt}
                if not by:
                    continue
                u, p = by.get("unpruned", {}), by.get("pruned", {})
                if u.get("status") != "ok" or p.get("status") != "ok":
                    reason = "; ".join("%s: %s" % (k, v.get("reason")) for k, v in by.items()
                                       if v.get("status") != "ok")
                    md.append("| %s | %s | - | - | - | - | - | - | - | not measured (%s) | - |" % (dev, rt, reason))
                    continue
                cu = np.asarray(u["predictions"]) == labels
                cp = np.asarray(p["predictions"]) == labels
                mc = mcnemar_exact(cp, cu)
                lu, lp = u["latency_raw_ms"], p["latency_raw_ms"]
                sp = float(np.median(lu) / np.median(lp))
                ci = bootstrap_ratio_of_medians(lu, lp)
                acc_u, acc_p = u["accuracy"]["top1"], p["accuracy"]["top1"]
                md.append("| %s | %s | %d | %.2f%% | %.2f%% | %+.2f | %.3g | %.3f | %.3f | %.3fx (%.3f-%.3f) | %d |" % (
                    dev, rt, cp.size, 100 * acc_u, 100 * acc_p, 100 * (acc_p - acc_u), mc["p_value"],
                    np.median(lu), np.median(lp), sp, ci[0], ci[1], r["settings"]["threads"]))
                fig_rows.append((r["device_label"], rt, float(np.median(lu)), float(np.median(lp))))
                se = sustained_energy(r, rt, power_logs) or {"models": {}, "source": None}
                for role, x in (("unpruned", u), ("pruned", p)):
                    sm = se["models"].get(role, {})
                    L = x["latency_ms"]
                    rows_csv.append({
                        "experiment": exp, "device": r["device_label"], "emulated": r["emulated"],
                        "runtime": rt, "runtime_version": x.get("runtime_version"), "model": role,
                        "params": r["models"][role]["params"], "macs": r["models"][role]["complexity"]["macs"],
                        "file_bytes": x["file_bytes"], "images": x["accuracy"]["n"],
                        "correct": x["accuracy"]["correct"], "top1": x["accuracy"]["top1"],
                        "top1_ci_lo": x["accuracy"]["wilson95"][0], "top1_ci_hi": x["accuracy"]["wilson95"][1],
                        "lat_median_ms": L["median"], "lat_mean_ms": L["mean"], "lat_std_ms": L["std"],
                        "lat_p95_ms": L["p95"], "lat_min_ms": L["min"], "lat_max_ms": L["max"],
                        "throughput_img_s": L["throughput_img_s_from_median"], "latency_runs": L["n"],
                        "top5": x["accuracy"].get("top5"),
                        "load_ms": (x.get("probe") or {}).get("median", {}).get("load_ms"),
                        "first_inference_ms": (x.get("probe") or {}).get("median", {}).get("first_inference_ms"),
                        "peak_rss_mib": ((x.get("probe") or {}).get("median", {}).get("peak_rss_bytes") or 0)
                        / 2 ** 20 or None,
                        "sustained_img_s": sm.get("throughput_img_s"),
                        "energy_mj_per_inference": sm.get("mj_per_inference"),
                        "energy_mj_per_inference_above_idle": sm.get("mj_per_inference_above_idle"),
                        "avg_power_w": sm.get("avg_power_w"), "energy_source": se["source"],
                        "threads": r["settings"]["threads"], "results_file": r["_file"]})

        md.append("\nSpeedup = unpruned median / pruned median latency (batch 1); CI by bootstrap. "
                  "McNemar compares per-image correctness of the two models on the same images.\n")

        def ok_pairs():
            for r in rs:
                for rt in ALL_RUNTIMES:
                    by = {x["model"]: x for x in r["runs"] if x["runtime"] == rt and x.get("status") == "ok"}
                    if "unpruned" in by and "pruned" in by:
                        yield r, rt, by["unpruned"], by["pruned"]

        if (rs[0].get("num_classes") or 0) > 10:
            md.append("## Top-5 accuracy\n")
            md.append("| Device | Runtime | Images | Unpruned top-5 | Pruned top-5 | Δ (pp) |")
            md.append("|---|---|---|---|---|---|")
            for r, rt, u, p in ok_pairs():
                if "top5" in u["accuracy"]:
                    md.append("| %s | %s | %d | %.2f%% | %.2f%% | %+.2f |" % (
                        r["device_label"], rt, u["accuracy"]["n"], 100 * u["accuracy"]["top5"],
                        100 * p["accuracy"]["top5"], 100 * (p["accuracy"]["top5"] - u["accuracy"]["top5"])))
            md.append("")

        mem_rows = [(r, rt, u, p) for r, rt, u, p in ok_pairs() if u.get("probe") and p.get("probe")]
        if mem_rows:
            md.append("## Start-up time and memory (unpruned / pruned, fresh process, median of %d)\n"
                      % mem_rows[0][0]["settings"].get("mem_probes", 0))
            md.append("| Device | Runtime | Model load (ms) | First inference (ms) | Peak RAM of the process (MiB) | "
                      "RAM above the bare runtime (MiB) |")
            md.append("|---|---|---|---|---|---|")
            for r, rt, u, p in mem_rows:
                mu, mp = u["probe"]["median"], p["probe"]["median"]
                mib = lambda d, k: d[k] / 2 ** 20 if d.get(k) is not None else None
                md.append("| %s | %s | %s | %s | %s | %s |" % (
                    r["device_label"], rt, _pair(mu.get("load_ms"), mp.get("load_ms"), "%.1f"),
                    _pair(mu.get("first_inference_ms"), mp.get("first_inference_ms"), "%.1f"),
                    _pair(mib(mu, "peak_rss_bytes"), mib(mp, "peak_rss_bytes"), "%.1f"),
                    _pair(mib(mu, "peak_above_runtime_bytes"), mib(mp, "peak_above_runtime_bytes"), "%.1f")))
            md.append("\nLoad = creating the session / interpreter (file already in the OS cache); first "
                      "inference = the first call after loading (cold start).\n")

        sus_rows = []
        for r, rt, u, p in ok_pairs():
            e = sustained_energy(r, rt, power_logs)
            if e:
                sus_rows.append((r, rt, e))
        if sus_rows:
            md.append("## Sustained inference, temperature and energy (unpruned / pruned)\n")
            md.append("| Device | Runtime | Seconds per model | Throughput (img/s) | Temperature (°C) | "
                      "Energy per inference (mJ) | Above idle (mJ) | Avg power (W) | Energy source |")
            md.append("|---|---|---|---|---|---|---|---|---|")
            for r, rt, e in sus_rows:
                u, p = e["models"].get("unpruned", {}), e["models"].get("pruned", {})
                md.append("| %s | %s | %.0f | %s | %s | %s | %s | %s | %s |" % (
                    r["device_label"], rt, r["settings"].get("sustained_seconds", 0),
                    _pair(u.get("throughput_img_s"), p.get("throughput_img_s"), "%.2f"),
                    ("%.1f-%.1f" % (e["temp_min_c"], e["temp_max_c"])) if e["temp_min_c"] is not None else "-",
                    _pair(u.get("mj_per_inference"), p.get("mj_per_inference"), "%.1f"),
                    _pair(u.get("mj_per_inference_above_idle"), p.get("mj_per_inference_above_idle"), "%.1f"),
                    _pair(u.get("avg_power_w"), p.get("avg_power_w"), "%.2f"), e["source"] or "none"))
            md.append("\nEach model runs continuously in two windows (A B B A). Energy per inference = energy "
                      "over the windows / inferences; 'above idle' subtracts the idle power measured just "
                      "before. RAPL = whole CPU package; Pi 5 PMIC = SoC and memory rails; an external USB "
                      "meter = the whole board.\n")
            for r, rt, e in sus_rows:
                if e["source"] is None:
                    warn.append("%s: %s has no energy source; add --power-csv %s=<meter log>"
                                % (exp, r["device_label"], r["device_label"]))

        md.append("## Agreement between runtimes (same model, same device, same images)\n")
        md.append("| Device | Model | Reference | Runtime | Top-1 agreement | Top-1 (ref -> runtime) |")
        md.append("|---|---|---|---|---|---|")
        for dev_label in sorted({r["device_label"] for r in rs}):
            rs_dev = sorted([r for r in rs if r["device_label"] == dev_label], key=lambda r: r["created"])
            for role in ("unpruned", "pruned"):
                ok = {}
                for r in rs_dev:
                    for x in r["runs"]:
                        if x["model"] == role and x.get("status") == "ok":
                            ok[x["runtime"]] = (x, r["eval"]["indices"])
                ref = "pytorch" if "pytorch" in ok else ("onnx" if "onnx" in ok else None)
                if ref is None:
                    continue
                xr, idx_ref = ok[ref]
                for rt, (x, idx) in ok.items():
                    if rt == ref:
                        continue
                    if idx != idx_ref:
                        warn.append("%s: %s %s and %s on %s used different images - not compared"
                                    % (exp, role, ref, rt, dev_label))
                        continue
                    ag = float(np.mean(np.asarray(x["predictions"]) == np.asarray(xr["predictions"])))
                    md.append("| %s | %s | %s | %s | %.2f%% | %.2f%% -> %.2f%% |" % (
                        dev_label, role, ref, rt, 100 * ag, 100 * xr["accuracy"]["top1"],
                        100 * x["accuracy"]["top1"]))
                    if rt != "tflite_int8" and ag < 0.995:
                        warn.append("%s: %s %s vs %s on %s agree on only %.2f%% of images (FP32 runtimes "
                                    "should match)" % (exp, role, rt, ref, dev_label, 100 * ag))

        md.append("\n## Sources\n")
        for r in rs:
            e = r["env"]
            md.append("- `%s`: device **%s**%s, %s, %s, %s logical CPUs, Python %s, threads %d, "
                      "%d warm-up + %d timed runs, %d images (%s split)%s" % (
                          r["_file"], r["device_label"], " (emulated)" if r["emulated"] else "",
                          e.get("board_model") or e.get("cpu_model") or "cpu unknown", e["machine"],
                          e["logical_cpus"], e["python"], r["settings"]["threads"], r["settings"]["warmup"],
                          r["settings"]["runs"], r["eval"]["n_used"], r["eval"]["split"],
                          " SUBSET" if r["eval"]["subset"] else ""))
        md.append("")

        if fig_rows and not a.no_figure:
            try:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
                lbl = ["%s\n%s" % (d, rt) for d, rt, _, _ in fig_rows]
                xs = np.arange(len(fig_rows))
                fig, ax = plt.subplots(figsize=(max(6, 1.2 * len(fig_rows)), 4.2))
                ax.bar(xs - 0.2, [u for _, _, u, _ in fig_rows], 0.4, label="unpruned", color="#9aa5b1")
                ax.bar(xs + 0.2, [p for _, _, _, p in fig_rows], 0.4, label="pruned", color="#2a6f97")
                ax.set_yscale("log")
                ax.set_ylabel("median latency, batch 1 (ms, log scale)")
                ax.set_xticks(xs)
                ax.set_xticklabels(lbl, fontsize=8)
                ax.set_title("%s: pruned vs unpruned latency" % exp)
                ax.legend(frameon=False)
                fig.tight_layout()
                fn = "%s_latency.png" % exp
                fig.savefig(os.path.join(a.out, fn), dpi=200)
                plt.close(fig)
                md.append("![latency](%s)\n" % fn)
            except ImportError:
                pass

    if warn:
        md.append("## Warnings\n")
        md += ["- " + w for w in sorted(set(warn))]
    with open(os.path.join(a.out, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    if rows_csv:
        with open(os.path.join(a.out, "report.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows_csv[0]))
            w.writeheader()
            w.writerows(rows_csv)
    log("wrote %s" % os.path.join(a.out, "report.md"))



_DEFAULT_GATES = {"trunk", "sdta", "heads", "mlp"}
_SHORT = {"edgenext_xx_small": "XXS", "edgenext_x_small": "XS", "edgenext_small": "S", "edgenext_base": "B"}


def _method_of(L):
    m = L["command"]
    tags = []
    if m == "gated":
        gt = set((L.get("gates") or {}).get("types") or _DEFAULT_GATES)
        if gt != _DEFAULT_GATES:
            tags.append("gates=" + ",".join(sorted(gt)))
        rt = ((L.get("export") or {}).get("projection") or {}).get("round_to")
        if rt not in (None, 8):
            tags.append("round_to=%d" % rt)
    return m + ("[" + "; ".join(tags) + "]" if tags else "")


def _ms(vals, nd=2, scale=1.0, signed=False):
    v = [x * scale for x in vals if x is not None]
    if not v:
        return "-"
    f = "%+.*f" if signed else "%.*f"
    if len(v) == 1:
        return f % (nd, v[0])
    return (f + " ± %.*f") % (nd, float(np.mean(v)), nd, float(np.std(v, ddof=1)))


def cmd_summary(a):
    """One table per question across every run: accuracy (mean ± std over
    seeds, paired difference to the same-seed baseline), complexity, training
    cost, and every device/runtime measurement, plus trade-off figures."""
    def expand(pats):
        out = []
        for pat in pats or []:
            out += glob.glob(pat, recursive=True)
        return sorted(set(out))
    log_files, res_files = expand(a.logs), expand(a.results)
    if not log_files:
        raise SystemExit("no log.json files")
    power_logs = {}
    for spec in a.power_csv or []:
        lbl, path = spec.split("=", 1)
        power_logs[lbl] = read_power_csv(path)
    os.makedirs(a.out, exist_ok=True)
    warn = []

    runs, by_sha = [], {}
    for f in log_files:
        L = read_json(f)
        if "final_eval" not in L or "checkpoint" not in L:
            warn.append("%s: unfinished run (no final_eval)" % f)
            continue
        budget = (L.get("budget") or {}).get("fraction")
        rec = {"file": f, "dataset": L["data"]["dataset"], "arch": L["arch"], "method": _method_of(L),
               "budget": 1.0 if L["command"] == "baseline" else budget, "seed": L["seed"],
               "top1": L["final_eval"]["top1"], "top5": L["final_eval"].get("top5"),
               "n_eval": L["final_eval"]["n"], "params": L["model"]["params"], "macs": L["model"]["macs"],
               "hours": L.get("wall_time_s", 0) / 3600.0, "sha": L["checkpoint"]["sha256"],
               "amp": L.get("amp")}
        if L["command"] == "gated":
            ex = L.get("export", {})
            rec["closed"] = ex.get("units_closed_by_budget_projection")
            if not ex.get("cost_model_equals_measured", True) or ex.get("gated_vs_exported_max_abs_logit_diff", 0) > 1e-4:
                warn.append("%s: export check failed (cost model / gated-vs-exported)" % f)
            if not ex.get("budget_met", True):
                warn.append("%s: exported model is over budget" % f)
        if L["command"] == "depgraph" and not (L.get("depgraph") or {}).get("budget_reachable", True):
            warn.append("%s: DepGraph could not reach budget %.2f (reached %.1f%% of MACs)" % (
                f, budget, 100 * L["depgraph"]["macs"] / L["budget"]["unpruned_macs"]))
        runs.append(rec)
        by_sha[rec["sha"]] = rec
    base = {(r["dataset"], r["arch"], r["seed"]): r for r in runs if r["method"] == "baseline"}

    groups = {}
    for r in runs:
        groups.setdefault((r["dataset"], r["arch"], r["method"], r["budget"]), []).append(r)
    order = sorted(groups, key=lambda k: (k[0], k[1], k[2] != "baseline", k[2], -(k[3] or 0)))

    md = ["# Summary of all runs\n",
          "Computed from %d training logs and %d results files; nothing is entered by hand. "
          "Mean ± standard deviation over seeds; Δ = paired difference to the baseline of the same "
          "dataset, model and seed.\n" % (len(log_files), len(res_files))]
    md.append("## Accuracy, complexity and training cost (PyTorch, full evaluation split)\n")
    md.append("| Dataset | Model | Method | MAC budget | Seeds | Params (M) | MACs (M) | Top-1 (%) | Δ top-1 (pp) | "
              "Top-5 (%) | Train time (h) |")
    md.append("|---|---|---|---|---|---|---|---|---|---|---|")
    rows_csv = []
    for k in order:
        rs = groups[k]
        deltas = [100 * (r["top1"] - base[(r["dataset"], r["arch"], r["seed"])]["top1"]) for r in rs
                  if (r["dataset"], r["arch"], r["seed"]) in base and r["method"] != "baseline"]
        md.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            k[0], _SHORT.get(k[1], k[1]), k[2], "-" if k[2] == "baseline" else "%.2f" % k[3],
            ",".join(str(r["seed"]) for r in sorted(rs, key=lambda r: r["seed"])),
            _ms([r["params"] for r in rs], 3, 1e-6), _ms([r["macs"] for r in rs], 1, 1e-6),
            _ms([r["top1"] for r in rs], 2, 100), _ms(deltas, 2, signed=True) if deltas else "-",
            _ms([r["top5"] for r in rs], 2, 100), _ms([r["hours"] for r in rs], 2)))
        if len(rs) < 3:
            warn.append("%s %s %s budget %s: %d seed(s)" % (k[0], _SHORT.get(k[1], k[1]), k[2], k[3], len(rs)))
        if len({r["amp"] for r in rs + [base.get((r["dataset"], r["arch"], r["seed"])) or r for r in rs]}) > 1:
            warn.append("%s %s %s: --amp differs from its baseline" % (k[0], k[1], k[2]))
        rows_csv.append({"table": "accuracy", "dataset": k[0], "arch": k[1], "method": k[2], "budget": k[3],
                         "seeds": len(rs), "params_mean": np.mean([r["params"] for r in rs]),
                         "macs_mean": np.mean([r["macs"] for r in rs]),
                         "top1_mean": np.mean([r["top1"] for r in rs]),
                         "top1_std": np.std([r["top1"] for r in rs], ddof=1) if len(rs) > 1 else None,
                         "delta_top1_pp_mean": np.mean(deltas) if deltas else None,
                         "train_hours_mean": np.mean([r["hours"] for r in rs])})

    dev = {}
    for f in res_files:
        R = read_json(f)
        src = (R["models"].get("pruned") or {}).get("source_sha256")
        run = by_sha.get(src)
        if run is None:
            warn.append("%s: its pruned model matches no training log (pass its log.json)" % os.path.basename(f))
            continue
        for rt in ALL_RUNTIMES:
            by = {x["model"]: x for x in R["runs"] if x["runtime"] == rt and x.get("status") == "ok"}
            if "pruned" not in by or "unpruned" not in by:
                continue
            u, p = by["unpruned"], by["pruned"]
            se = sustained_energy(R, rt, power_logs) or {}
            sm = (se.get("models") or {})
            key = (run["dataset"], run["arch"], run["method"], run["budget"], R["device_label"], rt)
            dev.setdefault(key, []).append({
                "seed": run["seed"], "emulated": R["emulated"], "n": p["accuracy"]["n"],
                "top1_u": u["accuracy"]["top1"], "top1_p": p["accuracy"]["top1"],
                "lat_u": u["latency_ms"]["median"], "lat_p": p["latency_ms"]["median"],
                "speedup": u["latency_ms"]["median"] / p["latency_ms"]["median"],
                "ram_p": ((p.get("probe") or {}).get("median") or {}).get("peak_rss_bytes"),
                "ram_u": ((u.get("probe") or {}).get("median") or {}).get("peak_rss_bytes"),
                "first_p": ((p.get("probe") or {}).get("median") or {}).get("first_inference_ms"),
                "mj_p": (sm.get("pruned") or {}).get("mj_per_inference"),
                "mj_u": (sm.get("unpruned") or {}).get("mj_per_inference"),
                "tput_p": (sm.get("pruned") or {}).get("throughput_img_s")})
    if dev:
        md.append("\n## Edge-device measurements (batch 1; unpruned → pruned)\n")
        md.append("| Dataset | Model | Method | MAC budget | Device | Runtime | Seeds | Images | Top-1 (%) | "
                  "Median latency (ms) | Speedup | Peak RAM (MiB) | First inference (ms) | Energy (mJ/inference) |")
        md.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for key in sorted(dev, key=lambda k: (k[0], k[1], k[2], -(k[3] or 0), k[4], k[5])):
            xs = dev[key]
            fmt2 = lambda us, ps, nd, sc=1.0: "%s → %s" % (_ms(us, nd, sc), _ms(ps, nd, sc))
            md.append("| %s | %s | %s | %.2f | %s%s | %s | %d | %d | %s | %s | %s | %s | %s | %s |" % (
                key[0], _SHORT.get(key[1], key[1]), key[2], key[3], key[4],
                " (emulated)" if xs[0]["emulated"] else "", key[5], len(xs), xs[0]["n"],
                fmt2([x["top1_u"] for x in xs], [x["top1_p"] for x in xs], 2, 100),
                fmt2([x["lat_u"] for x in xs], [x["lat_p"] for x in xs], 2),
                _ms([x["speedup"] for x in xs], 2) + "x",
                fmt2([x["ram_u"] for x in xs], [x["ram_p"] for x in xs], 1, 2 ** -20),
                _ms([x["first_p"] for x in xs], 1),
                fmt2([x["mj_u"] for x in xs], [x["mj_p"] for x in xs], 1)))
            rows_csv.append({"table": "device", "dataset": key[0], "arch": key[1], "method": key[2],
                             "budget": key[3], "device": key[4], "runtime": key[5], "seeds": len(xs),
                             "latency_unpruned_ms": np.mean([x["lat_u"] for x in xs]),
                             "latency_pruned_ms": np.mean([x["lat_p"] for x in xs]),
                             "speedup_mean": np.mean([x["speedup"] for x in xs])})
        md.append("\nDevice accuracy is on the image count shown (a subset on slow devices); the table above "
                  "has the full-split accuracy. Peak RAM = whole process in a fresh run; energy needs an "
                  "on-board sensor or --power-csv.\n")

    if not a.no_figure:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            figs = []
            all_methods = sorted({k[2] for k in groups})
            fixed = {"baseline": "#222222", "gated": "#1f77b4", "depgraph": "#ff7f0e"}
            extra = ["#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#17becf", "#bcbd22", "#7f7f7f"]
            colour = {}
            for m in all_methods:
                colour[m] = fixed.get(m) or extra[len([c for c in colour if c not in fixed]) % len(extra)]
            for ds_arch in sorted({(k[0], k[1]) for k in groups}):
                axes_x = [("macs", "MACs (M)", None)]
                for key in sorted({(k[4], k[5]) for k in dev if (k[0], k[1]) == ds_arch}):
                    axes_x.append(("lat", "median latency, %s / %s (ms)" % key, key))
                fig, axs = plt.subplots(1, len(axes_x), figsize=(4.6 * len(axes_x), 3.8), squeeze=False)
                for ax, (kind, xl, dk) in zip(axs[0], axes_x):
                    methods = sorted({k[2] for k in groups if (k[0], k[1]) == ds_arch})
                    for m in methods:
                        pts = []
                        for k in groups:
                            if (k[0], k[1], k[2]) != (ds_arch[0], ds_arch[1], m):
                                continue
                            rs = groups[k]
                            y = [100 * r["top1"] for r in rs]
                            if kind == "macs":
                                x = [r["macs"] / 1e6 for r in rs]
                            else:
                                xs = dev.get((k[0], k[1], k[2], k[3]) + dk)
                                if not xs:
                                    if m == "baseline":
                                        others = [v for kk, v in dev.items() if (kk[0], kk[1]) == ds_arch
                                                  and kk[4:] == dk]
                                        x = [np.mean([e["lat_u"] for v in others for e in v])] if others else []
                                    else:
                                        x = []
                                else:
                                    x = [e["lat_p"] for e in xs]
                            if x:
                                pts.append((float(np.mean(x)), float(np.mean(y)),
                                            float(np.std(y, ddof=1)) if len(y) > 1 else 0.0))
                        if not pts:
                            continue
                        pts.sort()
                        mk = "*" if m == "baseline" else "o"
                        ax.errorbar([p[0] for p in pts], [p[1] for p in pts], yerr=[p[2] for p in pts],
                                    marker=mk, ms=10 if mk == "*" else 5, capsize=3, label=m, color=colour[m],
                                    linestyle="-" if len(pts) > 1 else "none")
                    ax.set_xlabel(xl)
                    ax.set_ylabel("top-1 (%)")
                    ax.grid(alpha=0.3)
                axs[0][0].legend(frameon=False, fontsize=8)
                fig.suptitle("%s, EdgeNeXt-%s: accuracy vs cost (mean ± std over seeds)"
                             % (ds_arch[0], _SHORT.get(ds_arch[1], ds_arch[1])))
                fig.tight_layout()
                fn = "tradeoff_%s_%s.png" % ds_arch
                fig.savefig(os.path.join(a.out, fn), dpi=200)
                plt.close(fig)
                figs.append(fn)
            if figs:
                md.append("## Trade-off figures\n")
                md += ["![%s](%s)\n" % (f, f) for f in figs]
        except ImportError:
            pass

    if warn:
        md.append("## Warnings\n")
        md += ["- " + w for w in sorted(set(warn))]
    with open(os.path.join(a.out, "summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    if rows_csv:
        keys = []
        for r in rows_csv:
            keys += [k for k in r if k not in keys]
        with open(os.path.join(a.out, "summary.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows_csv)
    log("wrote %s" % os.path.join(a.out, "summary.md"))



def _depgraph_failing_group(a):
    """Rebuild the dependency graph and list the first coupled group whose index
    list does not fit its own layers (the cause of the IndexError)."""
    out = ["", "DIAGNOSIS: first dependency group whose indices exceed a member's width"]
    try:
        import timm
        import torch
        import torch_pruning as tp
        model = timm.create_model(a.arch, pretrained=False, num_classes=a.num_classes).eval()
        dg = tp.DependencyGraph().build_dependency(
            model, example_inputs=torch.randn(1, 3, a.input_size, a.input_size))
        imp_cls = getattr(tp.importance, "GroupMagnitudeImportance", None) or tp.importance.MagnitudeImportance
        imp = imp_cls(p=2)
        names = {m: n for n, m in model.named_modules()}
        for g in dg.get_all_groups(ignored_layers=[model.get_classifier()],
                                   root_module_types=[torch.nn.Conv2d, torch.nn.Linear]):
            try:
                imp(g)
                continue
            except IndexError as e:
                out.append("group fails with: %s" % e)
            for dep, idxs in g:
                layer = dep.target.module
                w = getattr(layer, "weight", None)
                if w is None and not isinstance(layer, torch.nn.Parameter):
                    continue
                shape = tuple(w.shape) if w is not None else tuple(layer.shape)
                handler = getattr(dep.handler, "__name__", type(dep.handler).__name__)
                out.append("  %-50s %-12s shape=%-18s indices=%4d (max %d)  %s" % (
                    names.get(layer, "<parameter>"), type(layer).__name__, shape, len(idxs),
                    max(idxs) if len(idxs) else -1, handler))
            break
        else:
            out.append("  no failing group found")
    except Exception as e:
        out.append("  diagnosis unavailable: %s: %s" % (type(e).__name__, e))
    return out


def cmd_depgraph(a):
    import traceback
    import timm
    import torch
    lines = ["# DepGraph (Torch-Pruning) on timm EdgeNeXt", "created: " + now_iso()]
    env = env_info()
    lines += ["%s: %s" % (k, env["packages"].get(k)) for k in ("torch", "timm", "torch-pruning")]
    lines += ["python: " + env["python"], "arch: %s  num_classes: %d  pruning_ratio: %.2f  xca_patch: %s"
              % (a.arch, a.num_classes, a.ratio, a.patch_xca), ""]
    if a.patch_xca:
        install_xca_patch()
    status = "unknown"
    try:
        import torch_pruning as tp
        torch.manual_seed(0)
        model = timm.create_model(a.arch, pretrained=False, num_classes=a.num_classes).eval()
        ex = torch.randn(1, 3, a.input_size, a.input_size)
        before = count_params(model)
        imp_cls = getattr(tp.importance, "GroupMagnitudeImportance", None) or tp.importance.MagnitudeImportance
        pruner_cls = getattr(tp.pruner, "MetaPruner", None) or tp.pruner.BasePruner
        kw = dict(importance=imp_cls(p=2), ignored_layers=[model.get_classifier()])
        try:
            pruner = pruner_cls(model, ex, pruning_ratio=a.ratio, **kw)
        except TypeError:
            pruner = pruner_cls(model, ex, ch_sparsity=a.ratio, **kw)
        lines.append("stage: pruner.step()")
        pruner.step()
        lines.append("params: %d -> %d" % (before, count_params(model)))
        lines.append("stage: forward pass after pruning")
        with torch.inference_mode():
            out = model(ex)
        lines.append("forward OK, output shape %s" % (tuple(out.shape),))
        status = "succeeded"
    except Exception:
        status = "failed"
        lines += ["", "TRACEBACK:", traceback.format_exc()]
        lines += _depgraph_failing_group(a)
    lines.insert(2, "result: " + status)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    log("\n".join(lines))
    log("wrote " + a.out)



def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("export", help="package pruned + unpruned for all runtimes")
    e.add_argument("--experiment", required=True, help="short tag, e.g. c10_xxs")
    e.add_argument("--dataset", required=True, choices=list(DATASETS))
    e.add_argument("--data-root", required=True)
    e.add_argument("--pruned", required=True)
    e.add_argument("--unpruned", required=True)
    e.add_argument("--out", required=True)
    e.add_argument("--arch", help="override arch inference")
    e.add_argument("--input-size", type=int, default=224)
    e.add_argument("--calib-n", type=int, default=200, help="INT8 calibration images (train split)")
    e.add_argument("--class-order", help="tiny10/tiny200: comma-separated WNIDs in label order")
    e.add_argument("--eval-per-class", type=int,
                   help="cap the evaluation set at N images per class (imagefolder/ImageNet: "
                        "use 10 for laptops, 1-2 for a 1 GB Raspberry Pi); default: whole split")
    e.add_argument("--allow-arch-mismatch", action="store_true",
                   help="reference model may be another EdgeNeXt variant (e.g. pruned S vs XS)")
    e.add_argument("--opset", type=int, default=17)
    e.add_argument("--seed", type=int, default=0)
    e.add_argument("--skip-tflite", action="store_true")
    e.add_argument("--skip-int8", action="store_true")
    e.add_argument("--download", action="store_true")
    e.set_defaults(func=cmd_export)

    b = sub.add_parser("bench", help="measure accuracy + latency on this device")
    b.add_argument("--artifacts", required=True)
    b.add_argument("--device-label", required=True, help="e.g. laptop, rpi3bplus, qemu-a72")
    b.add_argument("--runtimes", help="comma list from %s (default: all)" % ",".join(ALL_RUNTIMES))
    b.add_argument("--threads", type=int, help="default min(4, cpu count); same for every runtime")
    b.add_argument("--warmup", type=int, default=20)
    b.add_argument("--runs", type=int, default=200, help="timed runs per model")
    b.add_argument("--rounds", type=int, default=10, help="interleaving rounds")
    b.add_argument("--limit", type=int, help="class-balanced subset of about N images (slow devices)")
    b.add_argument("--per-class", type=int, help="class-balanced subset, N per class")
    b.add_argument("--emulated", choices=["auto", "yes", "no"], default="auto")
    b.add_argument("--mem-probes", type=int, default=3,
                   help="fresh-process runs per model for load time, cold-start latency and peak RAM (0 = off)")
    b.add_argument("--sustained-seconds", type=float, default=0,
                   help="continuous-inference seconds per model for sustained throughput, temperature and "
                        "energy (e.g. 60; 0 = off)")
    b.add_argument("--idle-seconds", type=float, default=10, help="idle window before the sustained runs")
    b.add_argument("--progress", type=int, default=1000)
    b.add_argument("--out")
    b.set_defaults(func=cmd_bench)

    r = sub.add_parser("report", help="combine results into one report")
    r.add_argument("--results", nargs="+", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--no-figure", action="store_true")
    r.add_argument("--power-csv", nargs="*", metavar="DEVICE_LABEL=FILE",
                   help="external power-meter log for a device (columns: unix_time, watts or volts+amps)")
    r.set_defaults(func=cmd_report)

    sm = sub.add_parser("summary", help="all runs and seeds: accuracy, cost, device metrics, trade-off figures")
    sm.add_argument("--logs", nargs="+", required=True, help="log.json files of edgenext_gated_prune.py runs")
    sm.add_argument("--results", nargs="*", help="results/*.json files of bench runs")
    sm.add_argument("--out", required=True)
    sm.add_argument("--power-csv", nargs="*", metavar="DEVICE_LABEL=FILE")
    sm.add_argument("--no-figure", action="store_true")
    sm.set_defaults(func=cmd_summary)

    pp = sub.add_parser("probe", help="internal: one model in a fresh process (called by bench)")
    pp.add_argument("--runtime", required=True, choices=ALL_RUNTIMES)
    pp.add_argument("--model-file", required=True)
    pp.add_argument("--threads", type=int, default=1)
    pp.add_argument("--arch")
    pp.add_argument("--input", required=True)
    pp.add_argument("--steady", type=int, default=10)
    pp.set_defaults(func=cmd_probe)

    d = sub.add_parser("depgraph-repro", help="log what Torch-Pruning does on timm EdgeNeXt")
    d.add_argument("--arch", default="edgenext_xx_small")
    d.add_argument("--num-classes", type=int, default=10)
    d.add_argument("--ratio", type=float, default=0.5)
    d.add_argument("--input-size", type=int, default=224)
    d.add_argument("--patch-xca", action="store_true", help="install the XCA reshape patch first")
    d.add_argument("--out", default="depgraph_failure.log")
    d.set_defaults(func=cmd_depgraph)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
