#!/usr/bin/env bash
set -u
cd "$(dirname "$0")" || exit 1
HERE="$(pwd)"
LABEL="${LABEL:-rpi3bplus}"
SUSTAINED="${SUSTAINED:-60}"
VENV="${VENV:-$HOME/benchenv}"
STAMP="$(date +%Y%m%d-%H%M%S)"
LOG="$HERE/pi_bench_${LABEL}_${STAMP}.log"
exec > >(tee -a "$LOG") 2>&1

say()  { echo; echo "== $*"; }
fail() { echo; echo "STOPPED: $*"; echo "Send this log to your supervisor: $LOG"; exit 1; }

say "Raspberry Pi benchmark  ($(date))"
echo "folder : $HERE"
echo "label  : $LABEL"
echo "board  : $(tr -d '\0' 2>/dev/null < /proc/device-tree/model || echo unknown)"
echo "arch   : $(uname -m)"
echo "memory : $(free -m 2>/dev/null | awk '/Mem:/{print $2 " MB total, " $7 " MB available"}')"
echo "disk   : $(df -h "$HERE" | awk 'NR==2{print $4 " free"}')"

if [ "$(uname -m)" != "aarch64" ] && [ "${ALLOW_ANY_ARCH:-0}" != "1" ]; then
  fail "this needs 64-bit Raspberry Pi OS ('uname -m' must print aarch64). Reinstall the 64-bit OS with Raspberry Pi Imager."
fi
ZIPS=()
for z in *.zip; do
  [ -f "$z" ] || continue
  case "$z" in pi_results_*) continue;; esac
  ZIPS+=("$z")
done
if [ ${#ZIPS[@]} -eq 0 ] && ! ls -d */manifest.json >/dev/null 2>&1; then
  fail "no package .zip files in $HERE. Copy them next to this script."
fi

NEED=""
command -v unzip >/dev/null 2>&1 || NEED="$NEED unzip"
command -v zip   >/dev/null 2>&1 || NEED="$NEED zip"
python3 -c "import venv, ensurepip" >/dev/null 2>&1 || NEED="$NEED python3-venv"
if [ -n "$NEED" ]; then
  say "installing system packages:$NEED"
  sudo apt-get update && sudo apt-get install -y $NEED || fail "could not install:$NEED"
fi

if [ "${USE_SYSTEM_PYTHON:-0}" = "1" ]; then
  PY=python3
else
  if [ ! -x "$VENV/bin/python" ]; then
    say "creating Python environment in $VENV (first time only, ~5-10 min)"
    python3 -m venv "$VENV" || fail "python3 -m venv failed"
  fi
  PY="$VENV/bin/python"
  have_pkgs() {
    "$PY" -c "import numpy, PIL, onnxruntime" >/dev/null 2>&1 || return 1
    "$PY" -c "import ai_edge_litert.interpreter" >/dev/null 2>&1 && return 0
    "$PY" -c "import tflite_runtime.interpreter" >/dev/null 2>&1
  }
  if ! have_pkgs; then
    say "installing numpy, pillow, onnxruntime and the TFLite runtime"
    "$PY" -m pip install --upgrade pip >/dev/null
    "$PY" -m pip install numpy pillow onnxruntime || fail "pip install numpy pillow onnxruntime failed (check the internet connection)"
    "$PY" -m pip install ai-edge-litert || "$PY" -m pip install tflite-runtime || \
      fail "could not install a TFLite runtime (ai-edge-litert or tflite-runtime)"
  fi
fi
"$PY" - <<'PYEOF' || fail "Python packages are not usable"
import numpy, PIL, onnxruntime
try:
    import ai_edge_litert.interpreter as t; name = "ai-edge-litert"
except Exception:
    import tflite_runtime.interpreter as t; name = "tflite-runtime"
print("python packages OK: numpy", numpy.__version__, "| onnxruntime", onnxruntime.__version__, "|", name)
PYEOF

for z in "${ZIPS[@]}"; do
  if ! unzip -l "$z" | grep -q "manifest.json"; then
    echo "SKIPPING $z: no manifest.json inside (incomplete export - use the good copy)"; continue
  fi
  if ! unzip -l "$z" | grep -q "_int8.tflite"; then
    echo "SKIPPING $z: no TFLite files inside (this is the broken earlier export)"; continue
  fi
  dir=$(unzip -Z1 "$z" | grep 'manifest.json$' | head -1 | xargs dirname)
  if [ -f "$dir/manifest.json" ]; then
    echo "already unpacked: $dir"
  else
    say "unpacking $z"
    unzip -q -o "$z" || fail "could not unzip $z"
  fi
done

PKGS=$(ls -d */ 2>/dev/null | sed 's#/$##' | while read -r d; do [ -f "$d/manifest.json" ] && echo "$d"; done)
[ -n "$PKGS" ] || fail "no usable package folders found"
echo "packages: $(echo $PKGS)"

throttle() { command -v vcgencmd >/dev/null 2>&1 && vcgencmd get_throttled || echo "throttled=unknown (no vcgencmd)"; }
temp()     { command -v vcgencmd >/dev/null 2>&1 && vcgencmd measure_temp || echo "temp=unknown"; }

for p in $PKGS; do
  cd "$HERE/$p" || continue
  if [ "${RERUN:-0}" != "1" ] && ls results/${LABEL}__*.json >/dev/null 2>&1; then
    say "$p: results for $LABEL exist already - skipped (RERUN=1 to measure again)"; continue
  fi
  NC=$("$PY" -c "import json; print(json.load(open('manifest.json'))['dataset']['num_classes'])")
  if [ -n "${PER_CLASS:-}" ]; then PC="$PER_CLASS"
  else case "$NC" in 10) PC=100;; 100) PC=10;; 200) PC=5;; *) PC=1;; esac; fi
  say "$p: $NC classes, $PC images per class, sustained ${SUSTAINED}s  ($(date +%H:%M))"
  echo "before: $(throttle)  $(temp)"
  "$PY" edgenext_bench.py bench --artifacts . --device-label "$LABEL" \
      --runtimes onnx,tflite_fp32,tflite_int8 --per-class "$PC" \
      --sustained-seconds "$SUSTAINED" ${EXTRA:-} \
    || echo "WARNING: bench for $p ended with an error (see above); continuing with the next package"
  echo "after : $(throttle)  $(temp)"
  case "$(throttle)" in *0x0) ;; *unknown*) ;; *) echo "WARNING: the Pi was throttled (heat or weak power supply) - fix cooling/power and run again with RERUN=1";; esac
  "$PY" edgenext_bench.py report --results "results/*.json" --out report >/dev/null 2>&1 \
    && echo "report: $HERE/$p/report/report.md"
done

cd "$HERE" || exit 1
OUT="pi_results_${LABEL}_${STAMP}.zip"
FILES=""
for p in $PKGS; do
  [ -d "$p/results" ] && FILES="$FILES $p/results"
  [ -d "$p/report" ]  && FILES="$FILES $p/report"
done
say "packing results"
zip -q -r "$OUT" $FILES "$(basename "$LOG")" 2>/dev/null || zip -q -r "$OUT" $FILES
ls -lh "$OUT"
say "DONE ($(date)). Send this file to your supervisor:  $HERE/$OUT"
