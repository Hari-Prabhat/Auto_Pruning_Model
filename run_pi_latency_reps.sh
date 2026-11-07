#!/usr/bin/env bash
set -u
cd "$(dirname "$0")" || exit 1
HERE="$(pwd)"
LABEL="${LABEL:-rpi3bplus}"
REPS="${REPS:-5}"
RUNTIMES="${RUNTIMES:-onnx tflite_fp32 tflite_int8}"
RUNS="${RUNS:-200}"
COOL_C="${COOL_C:-50}"
COOL_MAX="${COOL_MAX:-300}"
VENV="${VENV:-$HOME/benchenv}"
STAMP="$(date +%Y%m%d-%H%M%S)"
OUTDIR="$HERE/latency_reps"
LOG="$HERE/pi_latreps_${LABEL}_${STAMP}.log"
mkdir -p "$OUTDIR"
exec > >(tee -a "$LOG") 2>&1

say()  { echo; echo "== $*"; }
fail() { echo; echo "STOPPED: $*"; echo "Send this log to your supervisor: $LOG"; exit 1; }

say "Raspberry Pi repeated latency sessions  ($(date))"
echo "folder   : $HERE"
echo "label    : $LABEL"
echo "board    : $(tr -d '\0' 2>/dev/null < /proc/device-tree/model || echo unknown)"
echo "arch     : $(uname -m)"
echo "governor : $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo unknown)"
echo "settings : REPS=$REPS RUNS=$RUNS RUNTIMES='$RUNTIMES' COOL_C=$COOL_C COOL_MAX=$COOL_MAX"

if [ "$(uname -m)" != "aarch64" ] && [ "${ALLOW_ANY_ARCH:-0}" != "1" ]; then
  fail "this needs 64-bit Raspberry Pi OS ('uname -m' must print aarch64)."
fi

if [ "${USE_SYSTEM_PYTHON:-0}" = "1" ]; then PY=python3; else PY="$VENV/bin/python"; fi
[ -x "$PY" ] || [ "$PY" = "python3" ] || fail "$VENV not found - run run_pi_bench.sh once first (it creates the Python environment)."
"$PY" - <<'PYEOF' || fail "Python packages are not usable - run run_pi_bench.sh once first"
import numpy, PIL, onnxruntime
try:
    import ai_edge_litert.interpreter as t; name = "ai-edge-litert"
except Exception:
    import tflite_runtime.interpreter as t; name = "tflite-runtime"
print("python   : numpy", numpy.__version__, "| onnxruntime", onnxruntime.__version__, "|", name)
PYEOF

[ $# -eq 0 ] && set -- "$HERE"
PKG_PATHS=()
PKG_NAMES=" "
add_pkg() {
  local d name
  d="$(cd "$1" && pwd)"; name="$(basename "$d")"
  [ -f "$d/edgenext_bench.py" ] || { echo "SKIPPING $d: no edgenext_bench.py inside"; return; }
  ls "$d"/models/*_int8.tflite >/dev/null 2>&1 || { echo "SKIPPING $d: no TFLite files (broken earlier export)"; return; }
  case "$PKG_NAMES" in *" $name "*) echo "SKIPPING $d: a package called $name was already found"; return;; esac
  PKG_PATHS+=("$d"); PKG_NAMES="$PKG_NAMES$name "
}
for arg in "$@"; do
  [ -d "$arg" ] || { echo "SKIPPING $arg: not a folder"; continue; }
  if [ -f "$arg/manifest.json" ]; then add_pkg "$arg"; continue; fi
  for d in "$arg"/*/; do [ -f "$d/manifest.json" ] && add_pkg "$d"; done
done
[ ${#PKG_PATHS[@]} -gt 0 ] || fail "no unpacked package folders found in: $* (unzip the packages or run run_pi_bench.sh first)"
NP=${#PKG_PATHS[@]}
NRT=$(echo $RUNTIMES | wc -w)
echo "packages : $NP"
for d in "${PKG_PATHS[@]}"; do echo "           $d"; done
echo "sessions : $((NP * NRT * REPS)) in total (about 1 min each plus cooling; roughly $((NP * NRT * REPS * 2 / 60 + 1)) h)"

throttle() { command -v vcgencmd >/dev/null 2>&1 && vcgencmd get_throttled | cut -d= -f2 || echo "unknown"; }
cpu_temp() { local t; t=$(cat /sys/class/thermal/thermal_zone0/temp 2>/dev/null || echo 0); echo $(( t / 1000 )); }
cool_down() {
  local t waited=0
  t=$(cpu_temp)
  while [ "$t" -gt "$COOL_C" ] && [ "$waited" -lt "$COOL_MAX" ]; do sleep 15; waited=$((waited + 15)); t=$(cpu_temp); done
  [ "$waited" -gt 0 ] && echo "  cooled ${waited} s to ${t} C"
  [ "$t" -gt "$COOL_C" ] && echo "  WARNING: still ${t} C after ${COOL_MAX} s - measuring anyway (a fan/heatsink would help)"
  return 0
}
now_flags() {
  local v
  command -v vcgencmd >/dev/null 2>&1 || return 0
  v=$(( $(vcgencmd get_throttled | cut -d= -f2) ))
  [ $((v & 0x1)) -ne 0 ] && echo -n "under-voltage "
  [ $((v & 0x2)) -ne 0 ] && echo -n "frequency-capped "
  [ $((v & 0x4)) -ne 0 ] && echo -n "throttled "
  [ $((v & 0x8)) -ne 0 ] && echo -n "soft-temperature-limit "
  return 0
}

DONE=0; FAILED=0; FLAGGED=0
for r in $(seq 1 "$REPS"); do
  say "repetition $r of $REPS  ($(date +%H:%M))"
  for i in $(seq 0 $((NP - 1))); do
    d="${PKG_PATHS[$(( (i + r - 1) % NP ))]}"; p="$(basename "$d")"
    for rt in $RUNTIMES; do
      OUTJ="$OUTDIR/$p/${LABEL}__${rt}__rep${r}.json"
      if [ -f "$OUTJ" ]; then DONE=$((DONE + 1)); continue; fi
      echo "-- $p / $rt / rep $r  ($(date +%H:%M:%S))"
      cool_down
      t0=$(cpu_temp)
      ( cd "$d" && "$PY" edgenext_bench.py bench --artifacts . --device-label "$LABEL" \
            --runtimes "$rt" --limit 20 --mem-probes 0 --sustained-seconds 0 \
            --runs "$RUNS" --rounds 10 --out "$OUTJ" ) > "$OUTDIR/last_session.txt" 2>&1
      rc=$?
      flags=$(now_flags)
      if [ $rc -ne 0 ] || [ ! -f "$OUTJ" ]; then
        FAILED=$((FAILED + 1)); echo "  ERROR (exit $rc) - last lines:"; tail -n 15 "$OUTDIR/last_session.txt"
        continue
      fi
      DONE=$((DONE + 1))
      med=$("$PY" -c "import json,sys; R=json.load(open(sys.argv[1])); print('  '.join('%s %.1f ms' % (r['model'], r['latency_ms']['median']) for r in R['runs'] if r.get('status') == 'ok'))" "$OUTJ" 2>/dev/null)
      echo "  ok  ${t0} C -> $(cpu_temp) C  throttled=$(throttle)  $med"
      if [ -n "$flags" ]; then FLAGGED=$((FLAGGED + 1)); echo "  WARNING: Pi reported $flags"; fi
    done
  done
done
rm -f "$OUTDIR/last_session.txt"

say "summary"
"$PY" - "$OUTDIR" <<'PYEOF' | tee "$OUTDIR/summary.txt"
import json, glob, os, sys, statistics as st
root = sys.argv[1]
rows = {}
sha = {}
pooled = {}
hot = 0; total = 0
for f in sorted(glob.glob(os.path.join(root, "*", "*__rep*.json"))):
    pkg = os.path.basename(os.path.dirname(f))
    R = json.load(open(f)); total += 1
    try:
        if R["runtime_info"][0]["state_after"]["throttled"]["soft_temp_limit_now"]: hot += 1
    except Exception:
        pass
    for run in R["runs"]:
        if run.get("status") != "ok": continue
        k = (pkg, run["runtime"], run["model"]); m = run["latency_ms"]["median"]
        rows.setdefault(k, []).append(m); sha[k] = run["file_sha256"]
        pooled.setdefault((run["runtime"], run["file_sha256"]), []).append(m)
print("sessions read: %d   (soft temperature limit active at the end of %d)" % (total, hot))
print("\nmedian latency per session (ms); value = median over sessions, [min-max]")
print("%-26s %-12s %-9s %3s %8s %17s" % ("package", "runtime", "model", "n", "median", "range"))
for k in sorted(rows):
    v = rows[k]
    print("%-26s %-12s %-9s %3d %8.2f   [%6.2f-%6.2f]" % (k[0], k[1], k[2], len(v), st.median(v), min(v), max(v)))
print("\nspeedup = unpruned / pruned, both medians over sessions; the unpruned model of a seed is")
print("pooled over every package that contains the same file (identical SHA-256)")
print("%-26s %-12s %8s %9s %8s" % ("package", "runtime", "pruned", "unpruned", "speedup"))
for k in sorted(rows):
    if k[2] != "pruned": continue
    ku = (k[0], k[1], "unpruned")
    if ku not in sha: continue
    u = st.median(pooled[(k[1], sha[ku])]); p = st.median(rows[k])
    print("%-26s %-12s %8.2f %9.2f %7.3fx" % (k[0], k[1], p, u, u / p))
PYEOF

cd "$HERE" || exit 1
OUT="pi_latreps_${LABEL}_${STAMP}.zip"
say "packing results"
zip -q -r "$OUT" latency_reps pi_latreps_*.log
ls -lh "$OUT"
say "DONE ($(date)): $DONE sessions ok, $FAILED failed, $FLAGGED with a throttling warning."
echo "Send this file to your supervisor:  $HERE/$OUT"
