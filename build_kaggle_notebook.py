"""build_kaggle_notebook.py - writes edgenext_kaggle.ipynb, the notebook
used to train and package every model of the paper on a Kaggle GPU.

The notebook embeds the current edgenext_bench.py and edgenext_gated_prune.py
(%%writefile cells), so rebuild it after changing either script:
    pip install nbformat
    python build_kaggle_notebook.py
    PRESET_DATASET=tiny200 PRESET_SEEDS="[0]" OUT_NAME=t200_s0.ipynb python build_kaggle_notebook.py
"""
import nbformat as nbf, os, sys
PRESET_DATASET = os.environ.get("PRESET_DATASET", "cifar10")
PRESET_SEEDS = os.environ.get("PRESET_SEEDS", "[0]")
OUT_NAME = os.environ.get("OUT_NAME", os.path.join(os.path.dirname(os.path.abspath(__file__)), "edgenext_kaggle.ipynb"))
ROOT = os.path.dirname(os.path.abspath(__file__))
B = open(os.path.join(ROOT, "edgenext_bench.py")).read()
P = open(os.path.join(ROOT, "edgenext_gated_prune.py")).read()
md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell
cells = [
md("""# EdgeNeXt pruning on Kaggle (training + packaging)

**Before running:** Settings (right panel) → Accelerator **GPU** (T4 or P100) and **Internet on** (needs a phone-verified account; used for pip, the pretrained weights and the datasets).

**Best way to run:** set the cell *Settings* below, then **Save Version → Save & Run All (Commit)**. It runs in the background (browser can be closed) for up to 12 h, and everything in `/kaggle/working` is saved as the version's output.

**Continue after a stop or for the next run:** *Add Input → Your Work → this notebook's output*. The *Restore* cell copies the earlier runs and `resume.pt` back, and `--resume` continues where it stopped. Runs that already have `log.json` are skipped.

Keep one commit under 12 h: EdgeNeXt-XXS (all three methods) fits in one; for EdgeNeXt-S run one method per commit.

`bench` is not run here (Kaggle is not an edge device)."""),
code(f"""# 1. Settings - change these for each commit
DATASET = '{PRESET_DATASET}'           # cifar10 | cifar100 | tiny200 | imagefolder
ARCH    = 'edgenext_xx_small'  # edgenext_xx_small | edgenext_small | edgenext_x_small
SEEDS   = {PRESET_SEEDS}                  # e.g. [0], [1, 2], [0, 1, 2]
BUDGET  = 0.5                  # 0.7 | 0.5 | 0.3 (fraction of unpruned MACs)
METHODS = ['baseline', 'gated', 'depgraph']   # S: one per commit, e.g. ['gated']
EXTRA_GATED = ''               # ablations, e.g. '--gate-types trunk,heads,mlp' (then also set TAG)
TAG     = ''                   # folder suffix for ablations, e.g. '-oldtrunk'
AMP     = ''                   # '' or '--amp' - the SAME for every run of a dataset
DO_TRAIN, DO_EXPORT = True, True   # export uses its own Python environment, so it can run any time
WORKERS = 4                    # Kaggle has 4 CPU cores"""),
code("""# 2. Work folder, GPU and internet
import os, glob, shutil, json, math, subprocess
WORK = '/kaggle/working/edgenext'
os.makedirs(WORK, exist_ok=True)
%cd {WORK}
!nvidia-smi --query-gpu=name,memory.total --format=csv
r = subprocess.run(['pip', 'download', '--no-deps', '-q', '-d', '/tmp/netcheck', 'six'], capture_output=True)
print('internet OK' if r.returncode == 0 else 'NO INTERNET: turn it on in Settings')"""),
code("""# 3. Restore earlier runs from attached outputs of this notebook (Add Input -> Your Work)
for src in glob.glob('/kaggle/input/*/edgenext') + glob.glob('/kaggle/input/*/*/edgenext'):
    for sub in ('runs', 'artifacts', 'evidence', 'reports'):
        s = os.path.join(src, sub)
        if os.path.isdir(s):
            shutil.copytree(s, os.path.join(WORK, sub), dirs_exist_ok=True)
            print('restored', s)
!ls -R {WORK}/runs 2>/dev/null | head -40"""),
code("%%writefile edgenext_bench.py\n" + B),
code("%%writefile edgenext_gated_prune.py\n" + P),
code("""# 6. Training packages and self-test (must end with ALL CHECKS PASSED)
!pip -q install timm==1.0.30 torch-pruning==1.6.1
!python -m py_compile edgenext_bench.py edgenext_gated_prune.py && echo "scripts OK"
!python edgenext_gated_prune.py check --archs {ARCH}"""),
code("""# 7. Data
DS  = {'cifar10': 'c10', 'cifar100': 'c100', 'tiny200': 't200', 'imagefolder': 'in1k'}[DATASET]
MD  = {'edgenext_xx_small': 'xxs', 'edgenext_small': 's', 'edgenext_x_small': 'xs'}[ARCH]
B   = int(round(BUDGET * 100))
RUN = f'{WORK}/runs/{DS}_{MD}'
os.makedirs('/kaggle/temp', exist_ok=True)   # scratch space, not saved as output
if DATASET in ('cifar10', 'cifar100'):
    DATA_ROOT = '/kaggle/temp/data'          # not saved as output (it is 170 MB)
    os.makedirs(DATA_ROOT, exist_ok=True)
    import torchvision
    C = torchvision.datasets.CIFAR10 if DATASET == 'cifar10' else torchvision.datasets.CIFAR100
    C(DATA_ROOT, train=True, download=True); C(DATA_ROOT, train=False, download=True)
elif DATASET == 'tiny200':
    found = [os.path.dirname(p) for p in glob.glob('/kaggle/input/**/wnids.txt', recursive=True)
             if os.path.exists(os.path.join(os.path.dirname(p), 'val', 'val_annotations.txt'))]
    if found:
        DATA_ROOT = found[0]                 # a Tiny-ImageNet dataset added with Add Input
    else:
        DATA_ROOT = '/kaggle/temp/tiny-imagenet-200'
        if not os.path.exists(f'{DATA_ROOT}/wnids.txt'):
            import zipfile
            z = '/kaggle/temp/tiny.zip'
            for url in ('https://cs231n.stanford.edu/tiny-imagenet-200.zip',
                        'http://cs231n.stanford.edu/tiny-imagenet-200.zip'):
                if os.path.exists(z):
                    os.remove(z)
                r = subprocess.run(['wget', '-q', '--tries=3', '--timeout=60', '-O', z, url])
                if r.returncode == 0 and zipfile.is_zipfile(z):
                    subprocess.run(['unzip', '-q', '-o', z, '-d', '/kaggle/temp'])
                    os.remove(z)
                    break
                print('download failed from', url)
            if os.path.exists(z):
                os.remove(z)
        if not os.path.exists(f'{DATA_ROOT}/wnids.txt'):
            raise SystemExit('Tiny-ImageNet is not available. Add Input -> Datasets -> search "tiny-imagenet-200", '
                             'add one that contains wnids.txt, train/ and val/val_annotations.txt, then run this cell again.')
else:
    DATA_ROOT = '/kaggle/temp/imagenet'      # needs train/<wnid>/*.JPEG and val/<wnid>/*.JPEG
print(DATASET, ARCH, SEEDS, BUDGET, 'data:', DATA_ROOT, '| runs:', RUN)
!ls {DATA_ROOT} | head"""),
code("""# 8. Training: every selected method and seed, one after another; finished runs are skipped
def out_dir(method, seed):
    if method == 'baseline':
        return f'{RUN}/baseline_s{seed}'
    return f'{RUN}/{method}{TAG if method == "gated" else ""}_b{B}_s{seed}'

if DO_TRAIN:
    for seed in SEEDS:
        for method in METHODS:
            out = out_dir(method, seed)
            if os.path.exists(f'{out}/log.json'):
                print('done already:', out); continue
            budget = '' if method == 'baseline' else f'--budget {BUDGET}'
            extra = EXTRA_GATED if method == 'gated' else ''
            print('\\n=== running', out)
            !python edgenext_gated_prune.py {method} --arch {ARCH} --dataset {DATASET} --data-root {DATA_ROOT} \\
                {budget} --seed {seed} {AMP} {extra} --workers {WORKERS} --resume --out {out}"""),
code("""# 9. Check every run before packaging
for seed in SEEDS:
    for method in METHODS:
        p = out_dir(method, seed) + '/log.json'
        if not os.path.exists(p):
            print(p, 'not finished'); continue
        L = json.load(open(p))
        line = (f"{os.path.basename(os.path.dirname(p))}: top-1 {100*L['final_eval']['top1']:.2f}% on "
                f"{L['final_eval']['n']} images, {L['model']['params']:,} params, {L['model']['macs']/1e6:.1f} M MACs, "
                f"{L['data']['n_train']} train images, {L.get('wall_time_s', 0)/3600:.2f} h")
        if 'export' in L:
            e = L['export']
            line += (f" | cost==measured {e['cost_model_equals_measured']}, gated-vs-exported "
                     f"{e['gated_vs_exported_max_abs_logit_diff']:.1e}, budget met {e['budget_met']}, "
                     f"closed by projection {e['units_closed_by_budget_projection']}")
        if 'depgraph' in L:
            line += f" | budget reachable {L['depgraph']['budget_reachable']}"
        if 'eval_before_finetune' in L:
            line += f" | before fine-tune {100*L['eval_before_finetune']['top1']:.2f}%"
        b = [h['train_loss'] for h in L['history'] if h['phase'] == 'B']
        if b:
            line += f" | fine-tune loss first/last {b[0]:.3f}/{b[-1]:.3f}"
            ref = L.get('phase_b_start_train_loss')    # loss when fine-tuning started
            if b[0] > 0.8 * math.log(L['data']['num_classes']) and (ref is None or ref < 0.5 * math.log(L['data']['num_classes'])):
                line += '  <-- COLLAPSED, rerun'
        print(line)"""),
code("""# 10. Export packages in a separate Python environment (Kaggle's preinstalled TensorFlow breaks
#     litert-torch), with the exact versions the scripts were tested with. CPU only; first time ~5-10 min.
EXPORT_ENV = '/kaggle/temp/exportenv'
EXPORT_PY = f'{EXPORT_ENV}/bin/python'
PINS = ('timm==1.0.30 onnx==1.23.2 onnxruntime==1.29.0 ai-edge-litert==2.3.0 litert-torch==0.10.0 '
        'ai-edge-quantizer==0.10.0 numpy pillow')
if DO_EXPORT and not os.path.exists(EXPORT_PY):
    r = subprocess.run(['python', '-m', 'venv', EXPORT_ENV])
    if r.returncode != 0 or not os.path.exists(EXPORT_PY):
        !pip -q install virtualenv && python -m virtualenv -q {EXPORT_ENV}
    !{EXPORT_PY} -m pip -q install --upgrade pip
    r = subprocess.run([EXPORT_PY, '-m', 'pip', '-q', 'install', 'torch==2.14.1', 'torchvision==0.29.1',
                        '--index-url', 'https://download.pytorch.org/whl/cpu'])
    if r.returncode != 0:                      # CPU index unavailable: same versions from PyPI
        !{EXPORT_PY} -m pip -q install torch==2.14.1 torchvision==0.29.1
    !{EXPORT_PY} -m pip -q install {PINS}
    !{EXPORT_PY} -c "import torch, timm, onnxruntime, litert_torch, ai_edge_quantizer; print('export env OK, torch', torch.__version__)"

if DO_EXPORT:
    extra = {'imagefolder': '--eval-per-class 10', 'cifar10': '--download', 'cifar100': '--download'}.get(DATASET, '')
    for seed in SEEDS:
        base = out_dir('baseline', seed) + '/model.pt'
        for method in [m for m in METHODS if m != 'baseline']:
            pruned = out_dir(method, seed) + '/model.pt'
            exp = os.path.basename(os.path.dirname(os.path.dirname(pruned))) + '_' + os.path.basename(os.path.dirname(pruned))
            dest = f'{WORK}/artifacts/{exp}'
            if not (os.path.exists(pruned) and os.path.exists(base)):
                print('skip', exp, '(pruned or baseline model missing)'); continue
            if os.path.exists(f'{dest}/manifest.json'):
                print('packaged already:', exp); continue
            for old in (dest, dest + '.zip'):          # remove a half-written earlier attempt
                if os.path.isdir(old): shutil.rmtree(old)
                elif os.path.exists(old): os.remove(old)
            !TQDM_DISABLE=1 {EXPORT_PY} edgenext_bench.py export --experiment {exp} --dataset {DATASET} \\
                --data-root {DATA_ROOT} --pruned {pruned} --unpruned {base} --out {dest} {extra}
            man = f'{dest}/manifest.json'
            print(exp, (json.load(open(man))['warnings'] or 'no warnings') if os.path.exists(man) else 'EXPORT FAILED - see above')
    for d in sorted(glob.glob(f'{WORK}/artifacts/*/')):     # zip only finished packages
        d = d.rstrip('/')
        man = d + '/manifest.json'
        if os.path.exists(man) and (not os.path.exists(d + '.zip') or os.path.getmtime(d + '.zip') < os.path.getmtime(man)):
            shutil.make_archive(d, 'zip', os.path.dirname(d), os.path.basename(d))
    !ls -lh {WORK}/artifacts/*.zip 2>/dev/null

# one zip with this dataset/model's run folders (log.json, model.pt): download it with the packages
if os.path.isdir(RUN):
    shutil.make_archive(f'{WORK}/runs_{DS}_{MD}', 'zip', f'{WORK}/runs', os.path.basename(RUN))
    print('run folders:', f'{WORK}/runs_{DS}_{MD}.zip', round(os.path.getsize(f'{WORK}/runs_{DS}_{MD}.zip') / 2**20, 1), 'MiB')"""),
code("""# 11. (once, after all runs) DepGraph failure evidence and the summary of every run
# !python edgenext_bench.py depgraph-repro --arch edgenext_xx_small --out {WORK}/evidence/depgraph_xxs.log
# !python edgenext_bench.py depgraph-repro --arch edgenext_small --out {WORK}/evidence/depgraph_s.log
# !python edgenext_bench.py summary --logs "{WORK}/runs/**/log.json" --results "{WORK}/artifacts/*/results/*.json" --out {WORK}/reports/summary
!rm -rf /kaggle/working/edgenext/__pycache__"""),
]
nb = nbf.v4.new_notebook()
nb.cells = cells
nb.metadata = {"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
               "language_info": {"name": "python"},
               "kaggle": {"accelerator": "gpu", "isInternetEnabled": True, "isGpuEnabled": True}}
nbf.write(nb, OUT_NAME)
