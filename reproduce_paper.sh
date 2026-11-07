#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")" || exit 1
TRAIN_PY="${TRAIN_PY:-python}"
run() { echo "+ $*"; if [ "${DRY_RUN:-0}" != "1" ]; then "$@"; fi; }

run "$TRAIN_PY" edgenext_gated_prune.py check --archs edgenext_xx_small

bash run_experiment.sh cifar10  0 1 2
bash run_experiment.sh cifar100 0 1 2
[ -f "${TINY_ROOT:-./data/tiny-imagenet-200}/wnids.txt" ] || [ "${DRY_RUN:-0}" = "1" ] || bash get_tiny_imagenet.sh
DATA_ROOT="${TINY_ROOT:-./data/tiny-imagenet-200}" bash run_experiment.sh tiny200 0 1 2

mkdir -p evidence/rerun
for arch in edgenext_xx_small edgenext_small; do
  run "$TRAIN_PY" edgenext_bench.py depgraph-repro --arch "$arch" --out "evidence/rerun/depgraph_${arch}.log" || true
  run "$TRAIN_PY" edgenext_bench.py depgraph-repro --arch "$arch" --patch-xca --out "evidence/rerun/depgraph_${arch}_xcapatch.log" || true
done

run "$TRAIN_PY" edgenext_bench.py summary --logs "runs/**/log.json" \
    --results "artifacts/*/results/*.json" --out reports/summary
echo "Next: copy artifacts/*.zip to the Raspberry Pi and follow README_raspberry_pi.md"
