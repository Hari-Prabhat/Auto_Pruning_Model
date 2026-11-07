"""INT8 numerical diagnostic: runs the same INT8 models with XNNPACK (4 and 1
threads), without XNNPACK (built-in kernels) and with TFLite reference kernels,
and dumps every intermediate tensor of one image. Results are compared with a
PC to find the operation that differs on this CPU."""
import sys, os, json, time, platform, numpy as np
pkgroot, kitdir, out = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(out, exist_ok=True)
CASES = [("c100_xxs_depgraph_b50_s1", "pruned"), ("c100_xxs_gated_b50_s1", "unpruned"), ("c100_xxs_gated_b50_s2", "unpruned")]
try:
    from ai_edge_litert.interpreter import Interpreter, OpResolverType
    import ai_edge_litert; rtname = "ai-edge-litert"
except Exception:
    from tflite_runtime.interpreter import Interpreter, OpResolverType; rtname = "tflite-runtime"
MODES = {"xnnpack4": (OpResolverType.AUTO, 4), "xnnpack1": (OpResolverType.AUTO, 1),
         "builtin": (OpResolverType.BUILTIN_WITHOUT_DEFAULT_DELEGATES, 4), "reference": (OpResolverType.BUILTIN_REF, 1)}
def mk(path, mode):
    rt, th = MODES[mode]
    it = Interpreter(model_path=path, num_threads=th, experimental_op_resolver_type=rt); it.allocate_tensors(); return it
def run(it, x):
    i = it.get_input_details()[0]
    if len(i["shape"]) == 4 and i["shape"][-1] == 3: x = x.transpose(0, 2, 3, 1)
    if np.issubdtype(i["dtype"], np.integer):
        s, z = i["quantization"]; x = np.clip(np.round(x / s + z), -128, 127)
    it.set_tensor(i["index"], np.ascontiguousarray(x.astype(i["dtype"]))); it.invoke()
info = {"platform": platform.platform(), "machine": platform.machine(), "python": sys.version.split()[0], "runtime": rtname,
        "numpy": np.__version__, "cases": {}}
try:
    from importlib.metadata import version; info["runtime_version"] = version(rtname)
except Exception:
    pass
for pkg, role in CASES:
    d = os.path.join(pkgroot, pkg)
    if not os.path.isdir(d):
        print("missing package folder", d); continue
    sys.path.insert(0, d)
    import importlib; eb = importlib.import_module("edgenext_bench"); importlib.reload(eb)
    sys.path.pop(0)
    cfg = json.load(open(os.path.join(d, "manifest.json")))["dataset"]["preprocess"]
    data = np.load(os.path.join(d, "evalset.npz")); imgs, labels = data["images"], data["labels"]
    idx = np.array([int(np.where(labels == c)[0][0]) for c in range(int(labels.max()) + 1)])
    X = eb.preprocess_batch(imgs[idx], cfg)
    model = os.path.join(d, "models", role + "_int8.tflite")
    rec = {"indices": idx.tolist(), "labels": labels[idx].tolist(), "modes": {}}
    for mode in MODES:
        n = 50 if mode == "reference" else len(idx)
        t0 = time.time(); it = mk(model, mode); o = it.get_output_details()[0]; preds, logits = [], []
        for k in range(n):
            run(it, X[k:k + 1]); y = it.get_tensor(o["index"]).astype(np.float32)
            s, z = o["quantization"]
            if s: y = (y - z) * s
            logits.append(y[0]); preds.append(int(np.argmax(y[0])))
        np.save(os.path.join(out, "%s__%s__%s_logits.npy" % (pkg, role, mode)), np.array(logits, np.float32))
        rec["modes"][mode] = {"n": n, "preds": preds, "seconds": round(time.time() - t0, 1),
                              "acc": float(np.mean(np.array(preds) == labels[idx][:n]))}
        print("%s %s %-9s n=%3d acc %.3f  %.0fs" % (pkg, role, mode, n, rec["modes"][mode]["acc"], time.time() - t0), flush=True)
    allout = os.path.join(kitdir, "%s__%s_int8_allout.tflite" % (pkg, role))
    if os.path.exists(allout):
        for mode in ("xnnpack4", "builtin"):
            it = mk(allout, mode); run(it, X[0:1])
            dump = {}
            for k, o in enumerate(it.get_output_details()):
                dump["t%03d" % k] = it.get_tensor(o["index"])
            np.savez_compressed(os.path.join(out, "%s__%s__%s_allout.npz" % (pkg, role, mode)), **dump)
            json.dump([o["name"] for o in it.get_output_details()], open(os.path.join(out, "%s__%s_allout_names.json" % (pkg, role)), "w"))
            print("  dumped %d tensors (%s)" % (len(dump), mode), flush=True)
    info["cases"]["%s__%s" % (pkg, role)] = rec
json.dump(info, open(os.path.join(out, "diag.json"), "w"), indent=1)
print("done:", out)
