#!/usr/bin/env bash
set -u
KIT="$(cd "$(dirname "$0")" && pwd)"
STAMP="$(date +%Y%m%d-%H%M%S)"
PKGROOT="${PKGROOT:-}"
if [ -z "$PKGROOT" ]; then
  for d in "$KIT/.." "$HOME/cifar100benchmark" "$HOME/benchmark_c100"; do
    if [ -f "$d/c100_xxs_depgraph_b50_s1/manifest.json" ]; then PKGROOT="$(cd "$d" && pwd)"; break; fi
  done
fi
[ -n "$PKGROOT" ] || { echo "STOPPED: cannot find the unpacked CIFAR-100 packages (c100_xxs_depgraph_b50_s1/ ...). Put the kit in that folder."; exit 1; }
if [ "${USE_SYSTEM_PYTHON:-0}" = "1" ]; then PY=python3; else PY="$HOME/benchenv/bin/python"; fi
[ -x "$PY" ] || [ "$PY" = "python3" ] || { echo "STOPPED: $HOME/benchenv not found (run run_pi_bench.sh once first)"; exit 1; }
OUT="$PKGROOT/pi_int8diag_$STAMP"
LOG="$OUT.log"
mkdir -p "$OUT"
{
  echo "== INT8 diagnostic  ($(date))"
  echo "packages: $PKGROOT"
  echo "board   : $(tr -d '\0' 2>/dev/null < /proc/device-tree/model || echo unknown)"
  echo "arch    : $(uname -m)"
  echo "cpu     : $(grep -m1 -E 'model name|Hardware' /proc/cpuinfo 2>/dev/null | cut -d: -f2)"
  grep -m1 Features /proc/cpuinfo 2>/dev/null
  "$PY" "$KIT/int8_diag.py" "$PKGROOT" "$KIT" "$OUT"
} 2>&1 | grep -v "^INFO:" | tee "$LOG"
cd "$PKGROOT" || exit 1
zip -q -r "pi_int8diag_$STAMP.zip" "$(basename "$OUT")" "$(basename "$LOG")"
ls -lh "pi_int8diag_$STAMP.zip"
echo "DONE. Send this file to your supervisor:  $PKGROOT/pi_int8diag_$STAMP.zip"
