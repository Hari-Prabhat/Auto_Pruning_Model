#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")" || exit 1
ROOT="$(pwd)"

DATASET="${1:-}"
shift || true
SEEDS="${*:-0}"
case "$DATASET" in
  cifar10)  DS=c10;  EXPORT_EXTRA="--download"; TRAIN_EXTRA="--download" ;;
  cifar100) DS=c100; EXPORT_EXTRA="--download"; TRAIN_EXTRA="--download" ;;
  tiny200)  DS=t200; EXPORT_EXTRA="";           TRAIN_EXTRA="" ;;
  *) echo "usage: bash run_experiment.sh <cifar10|cifar100|tiny200> [seeds ...]"; exit 1 ;;
esac

ARCH="${ARCH:-edgenext_xx_small}"
case "$ARCH" in
  edgenext_xx_small) MD=xxs ;; edgenext_small) MD=s ;; edgenext_x_small) MD=xs ;;
  *) echo "unknown ARCH $ARCH"; exit 1 ;;
esac
BUDGET="${BUDGET:-0.5}"
B=$(awk -v b="$BUDGET" 'BEGIN { printf "%d", b * 100 + 0.5 }')
METHODS="${METHODS:-baseline gated depgraph}"
WORKERS="${WORKERS:-4}"
TRAIN_PY="${TRAIN_PY:-python}"
EXPORT_PY="${EXPORT_PY:-python}"
if [ "$DATASET" = "tiny200" ]; then DATA_ROOT="${DATA_ROOT:-$ROOT/data/tiny-imagenet-200}"; else DATA_ROOT="${DATA_ROOT:-$ROOT/data}"; fi
RUN="$ROOT/runs/${DS}_${MD}"
ART="$ROOT/artifacts"

run() { echo "+ $*"; if [ "${DRY_RUN:-0}" != "1" ]; then "$@"; fi; }
out_dir() {
  if [ "$1" = "baseline" ]; then echo "$RUN/baseline_s$2"; else echo "$RUN/$1_b${B}_s$2"; fi
}

if [ "$DATASET" = "tiny200" ] && [ ! -f "$DATA_ROOT/wnids.txt" ] && [ "${DRY_RUN:-0}" != "1" ]; then
  echo "Tiny-ImageNet-200 not found in $DATA_ROOT (needs wnids.txt, train/ and val/val_annotations.txt)."
  echo "Run: bash get_tiny_imagenet.sh   or set DATA_ROOT."
  exit 1
fi
mkdir -p "$RUN" "$ART"
echo "dataset $DATASET | arch $ARCH | budget $BUDGET | seeds $SEEDS | data $DATA_ROOT | runs $RUN"

if [ "${DO_TRAIN:-1}" = "1" ]; then
  for seed in $SEEDS; do
    for method in $METHODS; do
      out=$(out_dir "$method" "$seed")
      if [ -f "$out/log.json" ]; then echo "done already: $out"; continue; fi
      budget_arg=()
      [ "$method" != "baseline" ] && budget_arg=(--budget "$BUDGET")
      echo; echo "=== running $out"
      run "$TRAIN_PY" edgenext_gated_prune.py "$method" --arch "$ARCH" --dataset "$DATASET" \
          --data-root "$DATA_ROOT" ${budget_arg[@]+"${budget_arg[@]}"} --seed "$seed" --workers "$WORKERS" \
          $TRAIN_EXTRA --resume --out "$out"
    done
  done
fi

if [ "${DO_EXPORT:-1}" = "1" ]; then
  for seed in $SEEDS; do
    base="$(out_dir baseline "$seed")/model.pt"
    for method in $METHODS; do
      [ "$method" = "baseline" ] && continue
      pruned="$(out_dir "$method" "$seed")/model.pt"
      exp="${DS}_${MD}_${method}_b${B}_s${seed}"
      dest="$ART/$exp"
      if [ "${DRY_RUN:-0}" != "1" ] && { [ ! -f "$pruned" ] || [ ! -f "$base" ]; }; then
        echo "skip $exp (pruned or baseline model missing)"; continue
      fi
      if [ -f "$dest/manifest.json" ]; then echo "packaged already: $exp"; continue; fi
      [ "${DRY_RUN:-0}" != "1" ] && rm -rf "$dest" "$dest.zip"
      run env TQDM_DISABLE=1 "$EXPORT_PY" edgenext_bench.py export --experiment "$exp" \
          --dataset "$DATASET" --data-root "$DATA_ROOT" --pruned "$pruned" --unpruned "$base" \
          --out "$dest" $EXPORT_EXTRA
      run "$TRAIN_PY" -c "import shutil, sys; shutil.make_archive(sys.argv[1], 'zip', sys.argv[2], sys.argv[3])" \
          "$dest" "$ART" "$exp"
    done
  done
  echo; echo "packages for the Raspberry Pi:"; ls -lh "$ART"/*.zip 2>/dev/null || true
fi
