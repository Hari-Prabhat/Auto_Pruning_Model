"""
edgenext_gated_prune.py - coupling-typed gated structured pruning for timm
EdgeNeXt, plus the reference points a paper needs, all trained with one recipe.

Subcommands
-----------
  baseline   UNPRUNED model: phase A + phase B of the recipe, no gates.
  gated      The proposed method. Phase A trains the backbone together with
             hard-concrete (L0) gates under an augmented-Lagrangian MAC budget;
             the gates are then exported as a physically smaller timm model;
             phase B fine-tunes it.
  depgraph   DepGraph (Torch-Pruning, L2 group magnitude) restricted to the
             groups it can process on EdgeNeXt (MLP hidden units of every block
             and the stage-1 trunk). Phase A dense, one-shot prune to the same
             MAC budget, phase B fine-tune.
  check      Self-test without data (run it once on every machine): masked
             LayerNorm, gated model == exported model, cost model == measured
             MACs, rebuild from state_dict, optional ONNX / TFLite parity.

All training commands use the same data pipeline, epochs, optimiser settings
and the same single evaluation at the end (no model selection on the test
set). Each writes <out>/model.pt (portable {format, arch, num_classes,
state_dict}) and <out>/log.json. Feed model.pt files to
`edgenext_bench.py export`.

What changed with respect to the archived pruning scripts
---------------------------------------------------------
 1. All gates are sampled once per forward pass. The archived hooks drew a
    new random sample in every hook, so one stage's trunk gate had a
    different value in every block of the same forward pass.
 2. Evaluating the gated model uses the binary export mask (log_alpha > 0).
    The archived code evaluated with soft sigmoid values.
 3. LayerNorms on a pruned trunk (stem, downsample, block norms, norm_xca,
    head norm) compute their statistics over the kept channels only. In the
    archived code the pruned channels still entered the statistics, so the
    exported model computed a different function from the trained one.
 4. The trunk of the SDTA stages (stages 2-4) is gated and exported. The
    gate respects the coupling inside the split-depthwise cascade
    (sp = conv(sp + x_i) adds chunk i to chunk i-1 channel by channel): one
    "cascade" gate of the conv width, repeated for every chunk, plus a free
    gate for the pass-through chunk. The archived scripts gated these trunks
    during training but kept them full at export.
 5. The budget cost is the exact MAC count of the model the gates export
    (Conv2d + Linear + XCA matmuls, the definition of edgenext_bench.py
    count_macs), checked against a measurement. The archived proxy assumed
    7x7 kernels everywhere, ignored the stem, and counted savings that were
    never exported, so "budget met" did not mean the model got smaller.
 6. The per-branch (CAT) scalar gates are removed: they were never exported.
 7. Export copies every tensor, including the positional-encoding
    projection and all MLP biases, and is verified numerically against the
    gated model.
 8. Before export every kept width is rounded to a multiple of 8 (CPU
    kernels: arbitrary widths such as 17 or 21 erased most of the latency
    gain in ONNX Runtime) and, if the binary gates exceed the budget, the
    lowest-logit units per MAC saved are closed until the model fits. Both
    are logged (--round-to 1 disables rounding).
 9. Taylor sensitivities come from the previous steps (EMA) so each step
    needs one backward pass (no retain_graph).
10. No test-set evaluation during training; one evaluation before phase B
    (reported, not used for decisions) and one at the end.

Example (CIFAR-10, EdgeNeXt-XXS, 50 % MAC budget, seed 0)
--------------------------------------------------------
  python edgenext_gated_prune.py check
  python edgenext_gated_prune.py baseline --arch edgenext_xx_small --dataset cifar10 \\
      --data-root ./data --seed 0 --out runs/c10_xxs/unpruned_s0
  python edgenext_gated_prune.py gated --arch edgenext_xx_small --dataset cifar10 \\
      --data-root ./data --budget 0.5 --seed 0 --out runs/c10_xxs/gated_b50_s0
  python edgenext_gated_prune.py depgraph --arch edgenext_xx_small --dataset cifar10 \\
      --data-root ./data --budget 0.5 --seed 0 --out runs/c10_xxs/depgraph_b50_s0

Long runs can be continued after a disconnect with the same command plus
--resume.
"""

import argparse
import copy
import math
import os
import random
import sys
import time
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import edgenext_bench as eb

GAMMA, ZETA = -0.1, 1.1
GATE_TYPES = ("trunk", "sdta", "heads", "mlp")

RECIPES = {
    "xxs": {"epochs_a": 15, "epochs_b": 10, "batch_size": {"cifar10": 32, "cifar100": 32}, "batch_default": 64,
            "opt_a": "adam", "lr_a": 1e-4, "wd_a": 0.0, "cosine_a": False, "label_smoothing": 0.0,
            "clip_a": 0.0, "lr_b": 1e-2, "nesterov_b": False, "clip_b": 5.0, "warmup_b": 1.0, "strong_aug": False,
            "gate_lr": 1e-3, "warmup_epochs": 0, "eta": 0.5, "lambda_max": 50.0, "rho_start": 0.1,
            "rho_end": 20.0, "beta_start": 2.0 / 3.0, "beta_end": 0.1, "alpha": 1e-3, "budget": 0.5},
    "s": {"epochs_a": 30, "epochs_b": 20, "batch_size": {}, "batch_default": 64,
          "opt_a": "adamw", "lr_a": 5e-4, "wd_a": 5e-4, "cosine_a": True, "label_smoothing": 0.1,
          "clip_a": 1.0, "lr_b": 3e-2, "nesterov_b": True, "clip_b": 5.0, "warmup_b": 1.0, "strong_aug": True,
          "gate_lr": 1e-2, "warmup_epochs": 5, "eta": 0.3, "lambda_max": 30.0, "rho_start": 0.05,
          "rho_end": 15.0, "beta_start": 2.0 / 3.0, "beta_end": 0.1, "alpha": 1e-4, "budget": 0.7},
}


def log(msg):
    eb.log(msg)


def hc_sample(log_alpha, beta):
    import torch
    u = torch.rand_like(log_alpha).clamp(1e-6, 1 - 1e-6)
    s = torch.sigmoid((torch.log(u) - torch.log(1 - u) + log_alpha) / beta)
    return (s * (ZETA - GAMMA) + GAMMA).clamp(0.0, 1.0)


def hard_mask(log_alpha, min_keep=1):
    """Binary export mask: stretched sigmoid > 0.5, i.e. log_alpha > 0. At least
    min_keep units stay open (the highest logits)."""
    import torch
    la = log_alpha.detach()
    m = (la > 0).to(la.dtype)
    if int(m.sum()) < min_keep:
        m[torch.topk(la, min_keep).indices] = 1.0
    return m


def masked_layer_norm(x, mask, weight, bias, eps, dim):
    """LayerNorm over `dim` using only channels where mask > 0 (weighted by the
    mask when it is soft); output multiplied by the mask. With a binary mask
    this equals LayerNorm on the kept channels and zero elsewhere."""
    import torch
    dt = x.dtype
    x = x.float()
    shape = [1] * x.dim()
    shape[dim] = -1
    m = mask.float().view(shape)
    denom = mask.float().sum().clamp_min(1e-6)
    mean = (x * m).sum(dim, keepdim=True) / denom
    xc = x - mean
    var = (xc * xc * m).sum(dim, keepdim=True) / denom
    y = xc * torch.rsqrt(var + eps) * weight.float().view(shape) + bias.float().view(shape)
    return (y * m).to(dt)


class GatedEdgeNeXt:
    """Wraps a timm EdgeNeXt with coupling-typed gates (hooks + LayerNorm
    patches). Built on CPU, then moved with .to(device).

    Gate groups (one logit per unit, init_logit = open):
      trunk_s{s}            stage trunk of a stage without SDTA block  [C]
      casc_s{s}, tail_s{s}  stage trunk of an SDTA stage: cascade columns [w]
                            (repeated for each of the n conv chunks) and the
                            pass-through chunk [C - n*w]
      head_s{s}_b{b}        XCA heads of an SDTA block                 [H]
      mlp_s{s}_b{b}         MLP hidden units of every block            [E]
    """

    def __init__(self, model, arch, num_classes, input_size=224, gate_types=GATE_TYPES,
                 init_logit=2.0, sens_decay=0.9):
        import torch
        import torch.nn as nn
        self.model = model
        self.arch = arch
        self.num_classes = int(num_classes)
        self.gate_types = tuple(gate_types)
        self.beta = 2.0 / 3.0
        self.training = True
        self.sens_decay = sens_decay
        self.grad_scale = 1.0
        self.sens = {}
        self._z = {}
        self._trunk = [None] * len(model.stages)
        self._handles = []
        self._ln_patched = []
        self.stages = self._describe(input_size)
        self.gates = nn.ParameterDict()
        for s, st in enumerate(self.stages):
            if st["mode"] == "free" and "trunk" in self.gate_types:
                self.gates["trunk_s%d" % s] = nn.Parameter(torch.full((st["dim"],), init_logit))
            if st["mode"] == "sdta" and "sdta" in self.gate_types:
                self.gates["casc_s%d" % s] = nn.Parameter(torch.full((st["w"],), init_logit))
                self.gates["tail_s%d" % s] = nn.Parameter(torch.full((st["dim"] - st["n"] * st["w"],), init_logit))
            for b, bi in enumerate(st["blocks"]):
                if bi["type"] == "sdta" and "heads" in self.gate_types:
                    self.gates["head_s%d_b%d" % (s, b)] = nn.Parameter(torch.full((bi["H"],), init_logit))
                if "mlp" in self.gate_types:
                    self.gates["mlp_s%d_b%d" % (s, b)] = nn.Parameter(torch.full((bi["E"],), init_logit))
        self._install()


    def _describe(self, input_size):
        import torch
        import torch.nn as nn
        m = self.model
        shapes = {}
        hs = [st.register_forward_hook(lambda mod, i, o, s=s: shapes.__setitem__(s, tuple(o.shape)))
              for s, st in enumerate(m.stages)]
        was = m.training
        m.eval()
        with torch.no_grad():
            m(torch.zeros(1, 3, input_size, input_size, device=next(m.parameters()).device))
        m.train(was)
        for h in hs:
            h.remove()
        conv = m.stem[0]
        self.stem_k = int(conv.in_channels // conv.groups * conv.kernel_size[0] * conv.kernel_size[1])
        stages = []
        for s, st in enumerate(m.stages):
            _, C, H, W = shapes[s]
            info = {"dim": int(C), "N": int(H * W), "ds_k": None, "blocks": [], "mode": "free", "n": 0, "w": 0}
            if not isinstance(st.downsample, nn.Identity):
                dc = st.downsample[1]
                info["ds_k"] = int(dc.kernel_size[0] * dc.kernel_size[1])
            for blk in st.blocks:
                bi = {"E": int(blk.mlp.fc1.out_features)}
                if hasattr(blk, "xca"):
                    n, w = len(blk.convs), int(blk.convs[0].in_channels)
                    heads = int(blk.xca.num_heads)
                    bi.update(type="sdta", n=n, w=w, H=heads, hd=int(blk.xca.qkv.out_features) // 3 // heads,
                              ck2=int(blk.convs[0].kernel_size[0] * blk.convs[0].kernel_size[1]),
                              pos_in=int(blk.pos_embd.token_projection.in_channels) if blk.pos_embd is not None else 0)
                    if info["mode"] == "sdta" and (info["n"], info["w"]) != (n, w):
                        raise ValueError("SDTA blocks of stage %d disagree on their split" % s)
                    info.update(mode="sdta", n=n, w=w)
                else:
                    bi.update(type="conv", k2=int(blk.conv_dw.kernel_size[0] * blk.conv_dw.kernel_size[1]))
                info["blocks"].append(bi)
            stages.append(info)
        return stages


    def to(self, dev):
        self.model.to(dev)
        self.gates.to(dev)
        return self

    def train(self, mode=True):
        self.training = mode
        self.model.train(mode)
        return self

    def eval(self):
        return self.train(False)

    def parameters_backbone(self):
        return self.model.parameters()

    def parameters_gates(self):
        return self.gates.parameters()

    def state_dict(self):
        return {"model": self.model.state_dict(), "gates": {k: v.detach().cpu() for k, v in self.gates.items()},
                "sens": {k: v.detach().cpu() for k, v in self.sens.items()}}

    def load_state_dict(self, sd):
        import torch
        self.model.load_state_dict(sd["model"])
        with torch.no_grad():
            for k, v in sd["gates"].items():
                self.gates[k].copy_(v)
        dev = next(self.model.parameters()).device
        self.sens = {k: v.to(dev) for k, v in sd.get("sens", {}).items()}

    def __call__(self, x):
        self.sample()
        return self.model(x)


    def sample(self):
        z = {}
        for k, la in self.gates.items():
            z[k] = hc_sample(la, self.beta) if self.training else hard_mask(la)
        self._z = z
        self._trunk = [self._build_trunk(s, z) for s in range(len(self.stages))]

    def _build_trunk(self, s, z):
        import torch
        st = self.stages[s]
        if st["mode"] == "free":
            return z.get("trunk_s%d" % s)
        c = z.get("casc_s%d" % s)
        if c is None:
            return None
        return torch.cat([c] * st["n"] + [z["tail_s%d" % s]])

    def hard_masks(self):
        return {k: hard_mask(la).cpu() for k, la in self.gates.items()}


    def _patch_ln(self, mod, getter, dim):
        orig = type(mod).forward

        def fwd(this, x, getter=getter, dim=dim, orig=orig):
            m = getter()
            if m is None:
                return orig(this, x)
            return masked_layer_norm(x, m, this.weight, this.bias, this.eps, dim)
        mod.forward = types.MethodType(fwd, mod)
        self._ln_patched.append(mod)

    def _install(self):
        import torch.nn as nn
        m = self.model
        self._patch_ln(m.stem[1], lambda: self._trunk[0], 1)
        for s, st in enumerate(m.stages):
            if not isinstance(st.downsample, nn.Identity):
                self._patch_ln(st.downsample[0], lambda s=s: self._trunk[s - 1], 1)
                self._handles.append(st.downsample.register_forward_hook(self._trunk_hook(s)))
            for b, blk in enumerate(st.blocks):
                self._patch_ln(blk.norm, lambda s=s: self._trunk[s], -1)
                if hasattr(blk, "xca"):
                    self._patch_ln(blk.norm_xca, lambda s=s: self._trunk[s], -1)
                    self._handles.append(blk.xca.proj.register_forward_pre_hook(self._head_hook(s, b)))
                self._handles.append(blk.mlp.fc1.register_forward_hook(self._mlp_hook(s, b)))
                self._handles.append(blk.register_forward_hook(self._trunk_hook(s)))
        self._patch_ln(m.head.norm, lambda: self._trunk[-1], 1)

    def remove_hooks(self):
        for h in self._handles:
            h.remove()
        self._handles = []
        for mod in self._ln_patched:
            if "forward" in mod.__dict__:
                del mod.forward
        self._ln_patched = []

    def _trunk_hook(self, s):
        def hook(_m, _i, out):
            t = self._trunk[s]
            return out if t is None else out * t.view(1, -1, 1, 1).to(out.dtype)
        return hook

    def _head_hook(self, s, b):
        key = "head_s%d_b%d" % (s, b)
        H = self.stages[s]["blocks"][b]["H"]

        def hook(_m, inputs):
            z = self._z.get(key)
            if z is None:
                return None
            x = inputs[0]
            B, N, C = x.shape
            return ((x.view(B, N, H, C // H) * z.view(1, 1, H, 1).to(x.dtype)).reshape(B, N, C),)
        return hook

    def _mlp_hook(self, s, b):
        key = "mlp_s%d_b%d" % (s, b)

        def hook(_m, _i, out):
            z = self._z.get(key)
            if z is None:
                return out
            y = out * z.to(out.dtype)
            if self.training and y.requires_grad:
                act = y.detach()

                def grad_hook(g, act=act):
                    p = (act.float() * g.float() / self.grad_scale) ** 2
                    score = p.mean(dim=tuple(range(p.dim() - 1))).detach()
                    old = self.sens.get(key)
                    self.sens[key] = score if old is None else self.sens_decay * old + (1 - self.sens_decay) * score
                y.register_hook(grad_hook)
            return y
        return hook


    def cost_from_counts(self, c):
        """Exact MACs of the exported model given unit counts per gate group
        (python ints) or their soft sums (tensors, differentiable)."""
        T = []
        for s, st in enumerate(self.stages):
            if st["mode"] == "free" and ("trunk_s%d" % s) in c:
                T.append(c["trunk_s%d" % s])
            elif st["mode"] == "sdta" and ("casc_s%d" % s) in c:
                T.append(st["n"] * c["casc_s%d" % s] + c["tail_s%d" % s])
            else:
                T.append(st["dim"])
        total = self.stem_k * T[0] * self.stages[0]["N"]
        for s, st in enumerate(self.stages):
            N, Ts = st["N"], T[s]
            if s > 0:
                total = total + st["ds_k"] * T[s - 1] * Ts * N
            for b, bi in enumerate(st["blocks"]):
                E = c.get("mlp_s%d_b%d" % (s, b), bi["E"])
                total = total + 2 * N * Ts * E
                if bi["type"] == "conv":
                    total = total + bi["k2"] * Ts * N
                else:
                    Wc = c.get("casc_s%d" % s, bi["w"])
                    total = total + bi["n"] * bi["ck2"] * Wc * N
                    if bi["pos_in"]:
                        total = total + bi["pos_in"] * Ts * N
                    Hk = c.get("head_s%d_b%d" % (s, b), bi["H"])
                    total = total + 4 * N * Ts * Hk * bi["hd"] + 2 * Hk * bi["hd"] * bi["hd"] * N
        return total + T[-1] * self.num_classes

    def soft_cost(self):
        return self.cost_from_counts({k: v.sum() for k, v in self._z.items()})

    def hard_counts(self):
        return {k: int(v.sum()) for k, v in self.hard_masks().items()}

    def hard_cost(self):
        return int(self.cost_from_counts(self.hard_counts()))

    def full_cost(self):
        return int(self.cost_from_counts({k: int(v.numel()) for k, v in self.gates.items()}))

    def sens_reg(self):
        reg = 0.0
        for k, imp in self.sens.items():
            z = self._z.get(k)
            if z is not None:
                reg = reg + ((1 - z) * imp.to(z.device)).sum()
        return reg

    def fit_budget(self, budget_macs, round_to=8):
        """Make the binary gates exportable as an efficient model within the
        budget: (1) every group's kept count is rounded to the nearest multiple
        of round_to (or the group's full size; head counts are not rounded);
        (2) while over budget, the group whose next step down closes the
        lowest-logit units per MAC saved is reduced by one step; (3) slack left
        by (2) is given back to the highest-logit units. Kept units are always
        the highest-logit ones of their group. Returns units closed/opened."""
        import torch
        la = {k: v.detach().cpu().clone() for k, v in self.gates.items()}
        sizes = {k: int(v.numel()) for k, v in la.items()}
        order = {k: torch.argsort(v, descending=True) for k, v in la.items()}
        before = {k: int(v.sum()) for k, v in self.hard_masks().items()}
        r = max(1, int(round_to))

        def allowed(k):
            n = sizes[k]
            if k.startswith("head") or r == 1 or n < r:
                return list(range(1, n + 1))
            vals = list(range(r, n + 1, r))
            return vals if vals[-1] == n else vals + [n]

        al = {k: allowed(k) for k in la}
        counts = {k: min(al[k], key=lambda v, c=before[k]: (abs(v - c), -v)) for k in la}
        cost = self.cost_from_counts(counts)
        while cost > budget_macs:
            best = None
            for k in counts:
                lower = [v for v in al[k] if v < counts[k]]
                if not lower:
                    continue
                nv = lower[-1]
                trial = dict(counts)
                trial[k] = nv
                c2 = self.cost_from_counts(trial)
                saved = cost - c2
                if saved <= 0:
                    continue
                removed = order[k][nv:counts[k]]
                score = float(la[k][removed].sum()) / saved
                if best is None or score < best[0]:
                    best = (score, k, nv, c2)
            if best is None:
                break
            _, k, nv, cost = best
            counts[k] = nv
        target = min(budget_macs, self.cost_from_counts(before))
        while True:
            best = None
            for k in counts:
                higher = [v for v in al[k] if v > counts[k]]
                if not higher:
                    continue
                nv = higher[0]
                trial = dict(counts)
                trial[k] = nv
                c2 = self.cost_from_counts(trial)
                if c2 > target or c2 <= cost:
                    continue
                added = order[k][counts[k]:nv]
                score = float(la[k][added].sum()) / (c2 - cost)
                if best is None or score > best[0]:
                    best = (score, k, nv, c2)
            if best is None:
                break
            _, k, nv, cost = best
            counts[k] = nv
        closed = opened = 0
        with torch.no_grad():
            for k, g in self.gates.items():
                keep = order[k][:counts[k]]
                newla = torch.full_like(la[k], -10.0)
                newla[keep] = torch.clamp(la[k][keep], min=1.0)
                closed += max(0, before[k] - counts[k])
                opened += max(0, counts[k] - before[k])
                g.copy_(newla.to(g.device))
        return {"units_closed": int(closed), "units_opened": int(opened), "round_to": r,
                "cost_macs": int(self.hard_cost()), "fits": self.hard_cost() <= budget_macs}



def export_pruned(g):
    """Physically pruned timm EdgeNeXt (CPU, eval) equal to g in eval mode."""
    import timm
    import torch
    import torch.nn as nn
    eb.install_edgenext_patches()
    old = g.model
    masks = g.hard_masks()
    idx = {k: torch.nonzero(m).flatten() for k, m in masks.items()}
    trunk = []
    for s, st in enumerate(g.stages):
        if st["mode"] == "free" and ("trunk_s%d" % s) in idx:
            trunk.append(idx["trunk_s%d" % s])
        elif st["mode"] == "sdta" and ("casc_s%d" % s) in idx:
            c, t = idx["casc_s%d" % s], idx["tail_s%d" % s]
            trunk.append(torch.cat([c + i * st["w"] for i in range(st["n"])] + [t + st["n"] * st["w"]]))
        else:
            trunk.append(torch.arange(st["dim"]))
    dims = tuple(int(t.numel()) for t in trunk)
    new = timm.create_model(g.arch, pretrained=False, num_classes=g.num_classes, dims=dims)

    def P(t):
        return t.detach().cpu()

    def copy_vec(dst, src, ix):
        dst.data.copy_(P(src)[ix])

    def linear(src, rows, cols):
        w = P(src.weight)
        w = w[rows] if rows is not None else w
        w = w[:, cols] if cols is not None else w
        lin = nn.Linear(int(w.shape[1]), int(w.shape[0]), bias=src.bias is not None)
        lin.weight.data.copy_(w)
        if src.bias is not None:
            lin.bias.data.copy_(P(src.bias)[rows] if rows is not None else P(src.bias))
        return lin

    with torch.no_grad():
        i0 = trunk[0]
        new.stem[0].weight.copy_(P(old.stem[0].weight)[i0])
        if old.stem[0].bias is not None:
            copy_vec(new.stem[0].bias, old.stem[0].bias, i0)
        copy_vec(new.stem[1].weight, old.stem[1].weight, i0)
        copy_vec(new.stem[1].bias, old.stem[1].bias, i0)
        for s, (sn, so) in enumerate(zip(new.stages, old.stages)):
            it = trunk[s]
            if s > 0 and not isinstance(so.downsample, nn.Identity):
                ip = trunk[s - 1]
                copy_vec(sn.downsample[0].weight, so.downsample[0].weight, ip)
                copy_vec(sn.downsample[0].bias, so.downsample[0].bias, ip)
                sn.downsample[1].weight.copy_(P(so.downsample[1].weight)[it][:, ip])
                if so.downsample[1].bias is not None:
                    copy_vec(sn.downsample[1].bias, so.downsample[1].bias, it)
            for b, (bn, bo) in enumerate(zip(sn.blocks, so.blocks)):
                e = idx.get("mlp_s%d_b%d" % (s, b), torch.arange(bo.mlp.fc1.out_features))
                bn.mlp.fc1 = linear(bo.mlp.fc1, e, it)
                bn.mlp.fc2 = linear(bo.mlp.fc2, it, e)
                copy_vec(bn.norm.weight, bo.norm.weight, it)
                copy_vec(bn.norm.bias, bo.norm.bias, it)
                if bo.gamma is not None:
                    copy_vec(bn.gamma, bo.gamma, it)
                if not hasattr(bo, "xca"):
                    bn.conv_dw.weight.copy_(P(bo.conv_dw.weight)[it])
                    if bo.conv_dw.bias is not None:
                        copy_vec(bn.conv_dw.bias, bo.conv_dw.bias, it)
                    continue
                st = g.stages[s]
                ci = idx.get("casc_s%d" % s, torch.arange(st["w"]))
                convs = []
                for co in bo.convs:
                    k = int(co.kernel_size[0])
                    cv = nn.Conv2d(int(ci.numel()), int(ci.numel()), k, padding=k // 2,
                                   groups=int(ci.numel()), bias=co.bias is not None)
                    cv.weight.copy_(P(co.weight)[ci])
                    if co.bias is not None:
                        cv.bias.copy_(P(co.bias)[ci])
                    convs.append(cv)
                bn.convs = nn.ModuleList(convs)
                bn.width = int(ci.numel())
                if bo.pos_embd is not None:
                    tp_new, tp_old = bn.pos_embd.token_projection, bo.pos_embd.token_projection
                    tp_new.weight.copy_(P(tp_old.weight)[it])
                    copy_vec(tp_new.bias, tp_old.bias, it)
                copy_vec(bn.norm_xca.weight, bo.norm_xca.weight, it)
                copy_vec(bn.norm_xca.bias, bo.norm_xca.bias, it)
                if bo.gamma_xca is not None:
                    copy_vec(bn.gamma_xca, bo.gamma_xca, it)
                H = int(bo.xca.num_heads)
                inner = int(bo.xca.qkv.out_features) // 3
                hd = inner // H
                hk = idx.get("head_s%d_b%d" % (s, b), torch.arange(H))
                qsel = torch.cat([torch.arange(h * hd, (h + 1) * hd) for h in hk.tolist()])
                bn.xca.qkv = linear(bo.xca.qkv, torch.cat([qsel, qsel + inner, qsel + 2 * inner]), it)
                bn.xca.proj = linear(bo.xca.proj, it, qsel)
                bn.xca.num_heads = int(hk.numel())
                bn.xca.temperature = nn.Parameter(P(bo.xca.temperature)[hk].clone())
        il = trunk[-1]
        copy_vec(new.head.norm.weight, old.head.norm.weight, il)
        copy_vec(new.head.norm.bias, old.head.norm.bias, il)
        new.head.fc.weight.copy_(P(old.head.fc.weight)[:, il])
        new.head.fc.bias.copy_(P(old.head.fc.bias))
    return new.eval()


def compare_outputs(a, b, x):
    import torch
    with torch.no_grad():
        return float((a(x).float().cpu() - b(x).float().cpu()).abs().max())



def train_transform(dataset, size, strong):
    from torchvision import transforms as T
    norm = T.Normalize(eb.IMAGENET_MEAN, eb.IMAGENET_STD)
    if dataset in ("cifar10", "cifar100"):
        base = [T.Resize((size, size)), T.RandomCrop(size, padding=size // 8 if strong else 8),
                T.RandomHorizontalFlip()]
    elif dataset in ("tiny10", "tiny200"):
        base = [T.RandomResizedCrop(size, scale=(0.70, 1.0)), T.RandomHorizontalFlip()]
    else:
        base = [T.RandomResizedCrop(size), T.RandomHorizontalFlip()]
    if strong:
        return T.Compose(base + [T.RandAugment(num_ops=2, magnitude=9), T.ToTensor(),
                                 T.RandomErasing(p=0.25), norm])
    return T.Compose(base + [T.ToTensor(), norm])


class ArrayTrain:
    """uint8 HWC images in memory -> augmented tensors (module level so it can
    be pickled by DataLoader workers on every OS)."""
    def __init__(self, imgs, labels, tf):
        self.imgs, self.labels, self.tf = imgs, labels, tf

    def __len__(self):
        return int(len(self.labels))

    def __getitem__(self, i):
        from PIL import Image
        return self.tf(Image.fromarray(self.imgs[i])), int(self.labels[i])


class ArrayEval:
    def __init__(self, imgs, labels, cfg):
        self.imgs, self.labels, self.cfg = imgs, labels, cfg

    def __len__(self):
        return int(len(self.labels))

    def __getitem__(self, i):
        import torch
        return torch.from_numpy(eb.preprocess(self.imgs[i], self.cfg)), int(self.labels[i])


class PathTrain:
    def __init__(self, paths, labels, tf):
        self.paths, self.labels, self.tf = paths, labels, tf

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        from PIL import Image
        with Image.open(self.paths[i]) as im:
            return self.tf(im.convert("RGB")), int(self.labels[i])


class PathEval:
    def __init__(self, paths, labels, cfg):
        self.paths, self.labels, self.cfg = paths, labels, cfg

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        import torch
        return torch.from_numpy(eb.preprocess(eb.load_presized(self.paths[i], self.cfg), self.cfg)), \
            int(self.labels[i])


def build_data(a, rc):
    """(train_set, eval_set, info). Evaluation preprocessing is the function
    edgenext_bench.py uses on every device."""
    ds = eb.DATASETS[a.dataset]
    cfg = dict(ds["preprocess"])
    cfg["size"] = a.input_size
    tf = train_transform(a.dataset, a.input_size, rc["strong_aug"])
    class_order = a.class_order.split(",") if a.class_order else None
    if a.dataset == "imagefolder":
        tr_paths, tr_labels, names = eb.list_imagefolder(a.data_root, "train", a.train_per_class)
        ev_paths, ev_labels, _ = eb.list_imagefolder(a.data_root, "val", a.eval_per_class)
        train, evals = PathTrain(tr_paths, tr_labels, tf), PathEval(ev_paths, ev_labels, cfg)
    else:
        tr_imgs, tr_labels, names = eb.load_raw_split(a.dataset, a.data_root, "train", class_order,
                                                      a.download, per_class=a.train_per_class)
        if a.dataset in ("tiny10", "tiny200"):
            class_order = names
        ev_imgs, ev_labels, _ = eb.load_raw_split(a.dataset, a.data_root, ds["eval_split"], class_order,
                                                  a.download, per_class=a.eval_per_class)
        train, evals = ArrayTrain(tr_imgs, tr_labels, tf), ArrayEval(ev_imgs, ev_labels, cfg)
    info = {"dataset": a.dataset, "num_classes": eb.dataset_num_classes(a.dataset, a.data_root),
            "class_names": list(names) if names is not None else None, "eval_split": ds["eval_split"],
            "n_train": len(tr_labels), "n_eval": len(ev_labels), "train_per_class_cap": a.train_per_class,
            "eval_per_class_cap": a.eval_per_class, "eval_preprocess": cfg,
            "train_transform": repr(tf)}
    return train, evals, info


def _worker_init(wid):
    import torch
    s = torch.initial_seed() % 2 ** 32
    np.random.seed(s)
    random.seed(s)


def make_loader(dset, bs, workers, seed, shuffle, dev, drop_last):
    import torch
    g = torch.Generator()
    g.manual_seed(seed)
    return torch.utils.data.DataLoader(dset, batch_size=bs, shuffle=shuffle, drop_last=drop_last,
                                       num_workers=workers, pin_memory=dev.type == "cuda",
                                       worker_init_fn=_worker_init, generator=g)


def evaluate(model, loader, dev, amp):
    import torch
    model.eval()
    correct, correct5, n = 0, 0, 0
    with torch.inference_mode():
        for x, y in loader:
            x = x.to(dev, non_blocking=True)
            with torch.autocast(device_type=dev.type, enabled=amp):
                logits = model(x).float().cpu()
            top = logits.topk(min(5, logits.shape[1]), dim=1).indices
            correct += int((top[:, 0] == y).sum())
            correct5 += int((top == y[:, None]).any(1).sum())
            n += int(y.numel())
    return {"correct": correct, "n": n, "top1": correct / max(n, 1),
            "top5_correct": correct5, "top5": correct5 / max(n, 1)}



def seed_all(seed):
    import torch
    random.seed(seed)
    np.random.seed(seed % 2 ** 32)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def recipe_for(a):
    rc = dict(RECIPES[a.recipe or ("xxs" if a.arch == "edgenext_xx_small" else "s")])
    for kv in a.override or []:
        k, v = kv.split("=", 1)
        if k not in rc:
            raise SystemExit("unknown recipe key %r (keys: %s)" % (k, ", ".join(sorted(rc))))
        old = rc[k]
        rc[k] = (v.lower() in ("1", "true", "yes")) if isinstance(old, bool) else type(old)(v) \
            if not isinstance(old, dict) else old
    if a.epochs_a is not None:
        rc["epochs_a"] = a.epochs_a
    if a.epochs_b is not None:
        rc["epochs_b"] = a.epochs_b
    if getattr(a, "budget", None) is not None:
        rc["budget"] = a.budget
    if getattr(a, "warmup_epochs", None) is not None:
        rc["warmup_epochs"] = a.warmup_epochs
    rc["batch"] = a.batch_size or rc["batch_size"].get(a.dataset, rc["batch_default"])
    return rc


class Run:
    def __init__(self, out):
        self.out = os.path.abspath(out)
        os.makedirs(self.out, exist_ok=True)
        self.resume_path = os.path.join(self.out, "resume.pt")

    def save(self, state):
        import torch
        tmp = self.resume_path + ".tmp"
        torch.save(state, tmp)
        os.replace(tmp, self.resume_path)

    def load(self):
        import torch
        if os.path.exists(self.resume_path):
            return torch.load(self.resume_path, map_location="cpu", weights_only=False)
        return None

    def finish(self, keep):
        if not keep and os.path.exists(self.resume_path):
            os.remove(self.resume_path)


def common_log(a, rc, info, dev, command):
    return {"schema": "edgenext-prune/1", "command": command, "argv": sys.argv, "created": eb.now_iso(),
            "env": eb.env_info(), "script_sha256": eb.sha256_file(os.path.abspath(__file__)),
            "bench_sha256": eb.sha256_file(os.path.abspath(eb.__file__)),
            "arch": a.arch, "seed": a.seed, "pretrained": not a.no_pretrained, "device": str(dev),
            "amp": bool(a.amp), "recipe": a.recipe or ("xxs" if a.arch == "edgenext_xx_small" else "s"),
            "recipe_values": {k: v for k, v in rc.items() if k != "batch_size"}, "data": info, "history": []}


def save_model(model, arch, path):
    eb.save_portable_checkpoint(model.cpu().eval(), arch, path)
    return {"file": os.path.basename(path), "sha256": eb.sha256_file(path), "bytes": os.path.getsize(path)}


def complexity(model, input_size):
    m = copy.deepcopy(model).cpu().eval()
    return {"params": eb.count_params(m), "macs": eb.count_macs(m, input_size)["macs"],
            "arch_summary": eb.arch_summary(m)}


def setup(a, command):
    import torch
    eb.install_edgenext_patches()
    rc = recipe_for(a)
    dev = torch.device("cuda" if torch.cuda.is_available() and not a.cpu else "cpu")
    if a.amp and dev.type != "cuda":
        log("--amp ignored on CPU")
        a.amp = False
    seed_all(a.seed)
    train_set, eval_set, info = build_data(a, rc)
    log("%s: %s on %s, %d train / %d %s images, %d classes, batch %d, device %s"
        % (command, a.arch, a.dataset, info["n_train"], info["n_eval"], info["eval_split"],
           info["num_classes"], rc["batch"], dev))
    eval_loader = make_loader(eval_set, 128, a.workers, a.seed, False, dev, False)
    return rc, dev, train_set, eval_loader, info


def new_model(a, num_classes):
    import timm
    return timm.create_model(a.arch, pretrained=not a.no_pretrained, num_classes=num_classes)


def epoch_loader(train_set, rc, a, dev, phase, epoch):
    seed_all(a.seed * 1000 + (0 if phase == "A" else 500) + epoch)
    return make_loader(train_set, rc["batch"], a.workers, a.seed * 1000 + epoch, True, dev, True)


def train_epoch_dense(model, loader, opt, dev, ls, clip, scaler, amp, lr_fn=None, step0=0):
    import torch
    import torch.nn.functional as F
    model.train()
    tot, n = 0.0, 0
    for i, (x, y) in enumerate(loader):
        if lr_fn is not None:
            for g in opt.param_groups:
                g["lr"] = lr_fn(step0 + i)
        x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=dev.type, enabled=amp):
            loss = F.cross_entropy(model(x), y, label_smoothing=ls)
        scaler.scale(loss).backward()
        if clip > 0:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        scaler.step(opt)
        scaler.update()
        tot += float(loss.detach()) * x.size(0)
        n += x.size(0)
    return tot / max(n, 1)


def make_opt_a(params_groups, rc, epochs):
    import torch
    cls = torch.optim.Adam if rc["opt_a"] == "adam" else torch.optim.AdamW
    opt = cls(params_groups)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1), eta_min=rc["lr_a"] * 0.01) \
        if rc["cosine_a"] else None
    return opt, sched


def dense_phase_a(model, a, rc, dev, train_set, run, state, lg, arch):
    """Phase A without gates (baseline, depgraph). Resumable."""
    import torch
    opt, sched = make_opt_a([{"params": model.parameters(), "lr": rc["lr_a"], "weight_decay": rc["wd_a"]}],
                            rc, rc["epochs_a"])
    scaler = torch.amp.GradScaler("cuda", enabled=a.amp)
    start = 0
    if state and state.get("phase") == "A":
        opt.load_state_dict(state["opt"])
        if sched and state.get("sched"):
            sched.load_state_dict(state["sched"])
        scaler.load_state_dict(state["scaler"])
        start = state["epoch"]
    for e in range(start, rc["epochs_a"]):
        t0 = time.time()
        loss = train_epoch_dense(model, epoch_loader(train_set, rc, a, dev, "A", e), opt, dev,
                                 rc["label_smoothing"], rc["clip_a"], scaler, a.amp)
        if sched:
            sched.step()
        lg["history"].append({"phase": "A", "epoch": e + 1, "train_loss": loss, "seconds": time.time() - t0})
        log("[A %d/%d] loss %.4f (%.0fs)" % (e + 1, rc["epochs_a"], loss, time.time() - t0))
        run.save({"phase": "A", "epoch": e + 1, "model_sd": model.state_dict(), "arch": arch,
                  "opt": opt.state_dict(), "sched": sched.state_dict() if sched else None,
                  "scaler": scaler.state_dict(), "log": lg})


def start_train_loss(model, train_set, rc, a, dev, batches=20):
    """Mean training loss (eval mode, no update) on the first `batches` batches of a
    fixed-seed loader. Training batches are reseeded per epoch, so this extra pass
    does not change any later batch."""
    import torch
    import torch.nn.functional as F
    model.eval()
    loader = make_loader(train_set, rc["batch"], a.workers, a.seed * 1000 + 999, True, dev, True)
    tot, n = 0.0, 0
    with torch.inference_mode():
        for i, (x, y) in enumerate(loader):
            if i >= batches:
                break
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            with torch.autocast(device_type=dev.type, enabled=a.amp):
                out = model(x)
            tot += float(F.cross_entropy(out.float(), y)) * x.size(0)
            n += x.size(0)
    del loader
    return tot / max(n, 1)


def phase_b(model, a, rc, dev, train_set, run, state, lg, arch, extra=None):
    """Post-export fine-tune, identical for every method: SGD, linear warm-up
    for warmup_b epochs, then cosine to zero, per step. Resumable. Stops early
    if the first epoch collapses to chance level."""
    import torch
    opt = torch.optim.SGD(model.parameters(), lr=rc["lr_b"], momentum=0.9, weight_decay=5e-4,
                          nesterov=rc["nesterov_b"])
    scaler = torch.amp.GradScaler("cuda", enabled=a.amp)
    steps_per_epoch = max(1, len(train_set) // rc["batch"])
    total = max(1, rc["epochs_b"] * steps_per_epoch)
    warm = min(total, int(round(rc.get("warmup_b", 0.0) * steps_per_epoch)))

    def lr_fn(step):
        if step < warm:
            return rc["lr_b"] * (step + 1) / warm
        f = (step - warm) / max(1, total - warm)
        return 0.5 * rc["lr_b"] * (1.0 + math.cos(math.pi * min(f, 1.0)))

    start = 0
    if state and state.get("phase") == "B":
        opt.load_state_dict(state["opt"])
        scaler.load_state_dict(state["scaler"])
        start = state["epoch"]
    chance = math.log(lg["data"]["num_classes"])
    if start == 0:
        lg["phase_b_start_train_loss"] = start_train_loss(model, train_set, rc, a, dev)
        log("before fine-tune train loss %.4f (chance level ln %d = %.3f)"
            % (lg["phase_b_start_train_loss"], lg["data"]["num_classes"], chance))
    ref = lg.get("phase_b_start_train_loss")
    for e in range(start, rc["epochs_b"]):
        t0 = time.time()
        loss = train_epoch_dense(model, epoch_loader(train_set, rc, a, dev, "B", e), opt, dev,
                                 rc["label_smoothing"], rc["clip_b"], scaler, a.amp, lr_fn, e * steps_per_epoch)
        lg["history"].append({"phase": "B", "epoch": e + 1, "train_loss": loss, "seconds": time.time() - t0,
                              "lr_end": opt.param_groups[0]["lr"]})
        log("[B %d/%d] loss %.4f (%.0fs)" % (e + 1, rc["epochs_b"], loss, time.time() - t0))
        if not math.isfinite(loss) or (ref is not None and ref < 0.5 * chance and loss > 0.8 * chance):
            raise SystemExit("phase B collapsed: train loss %.3f is at chance level (ln %d = %.3f) although the "
                             "model entered fine-tuning with loss %s. Start this run again in a NEW --out folder "
                             "with a lower fine-tune learning rate for every method, e.g. --override lr_b=%g"
                             % (loss, lg["data"]["num_classes"], chance,
                                "%.3f" % ref if ref is not None else "n/a", rc["lr_b"] / 3))
        st = {"phase": "B", "epoch": e + 1, "model_sd": model.state_dict(), "arch": arch,
              "opt": opt.state_dict(), "scaler": scaler.state_dict(), "log": lg}
        st.update(extra or {})
        run.save(st)


def resume_state(a, run):
    if not a.resume:
        if os.path.exists(run.resume_path):
            raise SystemExit("%s exists: add --resume to continue that run, or use a new --out" % run.resume_path)
        return None
    st = run.load()
    if st is None:
        log("--resume: nothing to resume, starting from scratch")
    else:
        log("resuming phase %s after epoch %d" % (st["phase"], st["epoch"]))
    return st


def finish(a, run, model, arch, lg, eval_loader, dev, t_start):
    res = evaluate(model.to(dev), eval_loader, dev, a.amp)
    log("final %s top-1: %d/%d = %.2f%%, top-5 %.2f%%" % (lg["data"]["eval_split"], res["correct"], res["n"],
                                                           100 * res["top1"], 100 * res["top5"]))
    lg["final_eval"] = dict(res, split=lg["data"]["eval_split"])
    lg["model"] = complexity(model, a.input_size)
    lg["checkpoint"] = save_model(model, arch, os.path.join(run.out, "model.pt"))
    lg["wall_time_s"] = lg.get("wall_time_s", 0.0) + time.time() - t_start
    eb.write_json(os.path.join(run.out, "log.json"), lg)
    run.finish(a.keep_resume)
    log("wrote %s and %s" % (os.path.join(run.out, "model.pt"), os.path.join(run.out, "log.json")))



def cmd_baseline(a):
    t_start = time.time()
    rc, dev, train_set, eval_loader, info = setup(a, "baseline")
    run = Run(a.out)
    state = resume_state(a, run)
    if state:
        model, _ = eb.rebuild_from_state_dict(state["model_sd"], a.arch)
        lg = state["log"]
    else:
        model = new_model(a, info["num_classes"])
        lg = common_log(a, rc, info, dev, "baseline")
    model.to(dev)
    if not state or state["phase"] == "A":
        dense_phase_a(model, a, rc, dev, train_set, run, state, lg, a.arch)
        state = None
    phase_b(model, a, rc, dev, train_set, run, state, lg, a.arch)
    finish(a, run, model, a.arch, lg, eval_loader, dev, t_start)



def anneal(step, total, start, end, schedule="linear"):
    f = min(max(step / max(total, 1), 0.0), 1.0)
    return start * (end / start) ** f if schedule == "exp" else start + (end - start) * f


def build_gated(a, num_classes, pretrained):
    import timm
    base = timm.create_model(a.arch, pretrained=pretrained, num_classes=num_classes)
    g = GatedEdgeNeXt(base, a.arch, num_classes, a.input_size, a.gate_types.split(","), a.init_logit, a.sens_ema)
    measured = eb.count_macs(base, a.input_size)["macs"]
    if g.full_cost() != measured:
        raise RuntimeError("cost model %d != measured MACs %d" % (g.full_cost(), measured))
    return g


def cmd_gated(a):
    import torch
    import torch.nn.functional as F
    t_start = time.time()
    for t in a.gate_types.split(","):
        if t not in GATE_TYPES:
            raise SystemExit("unknown gate type %r (choose from %s)" % (t, ",".join(GATE_TYPES)))
    rc, dev, train_set, eval_loader, info = setup(a, "gated")
    run = Run(a.out)
    state = resume_state(a, run)
    nc = info["num_classes"]

    if state and state["phase"] == "B":
        lg = state["log"]
        pruned, _ = eb.rebuild_from_state_dict(state["model_sd"], a.arch)
        pruned.to(dev)
    else:
        g = build_gated(a, nc, pretrained=not a.no_pretrained and not state)
        full = g.full_cost()
        budget = rc["budget"] * full
        if state:
            g.load_state_dict(state["gated_sd"])
            lg = state["log"]
        else:
            lg = common_log(a, rc, info, dev, "gated")
            lg["gates"] = {"types": a.gate_types.split(","), "init_logit": a.init_logit, "sens_ema": a.sens_ema,
                           "groups": {k: int(v.numel()) for k, v in g.gates.items()}}
            lg["unpruned"] = complexity(g.model, a.input_size)
            lg["budget"] = {"fraction": rc["budget"], "unpruned_macs": full, "target_macs": budget}
            log("unpruned: %s params, %.2f M MACs; budget %.2f M MACs (%.0f%%)"
                % (f"{lg['unpruned']['params']:,}", full / 1e6, budget / 1e6, 100 * rc["budget"]))
        g.to(dev)
        opt, sched = make_opt_a([{"params": list(g.parameters_gates()), "lr": rc["gate_lr"], "weight_decay": 0.0},
                                 {"params": list(g.parameters_backbone()), "lr": rc["lr_a"],
                                  "weight_decay": rc["wd_a"]}], rc, rc["epochs_a"])
        scaler = torch.amp.GradScaler("cuda", enabled=a.amp)
        lam, step, start = 0.0, 0, 0
        rho = rc["rho_start"]
        n_batches = len(train_set) // rc["batch"]
        total = rc["epochs_a"] * n_batches
        if state:
            opt.load_state_dict(state["opt"])
            if sched and state.get("sched"):
                sched.load_state_dict(state["sched"])
            scaler.load_state_dict(state["scaler"])
            lam, step, start = state["lam"], state["step"], state["epoch"]

        for e in range(start, rc["epochs_a"]):
            t0 = time.time()
            g.train()
            apply_budget = e >= rc["warmup_epochs"]
            sums = {"task": 0.0, "cost": 0.0, "delta": 0.0, "n": 0}
            deltas = []
            for x, y in epoch_loader(train_set, rc, a, dev, "A", e):
                x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
                g.beta = anneal(step, total, rc["beta_start"], rc["beta_end"])
                rho = anneal(step, total, rc["rho_start"], rc["rho_end"], "exp")
                g.grad_scale = scaler.get_scale() if a.amp else 1.0
                opt.zero_grad(set_to_none=True)
                with torch.autocast(device_type=dev.type, enabled=a.amp):
                    task = F.cross_entropy(g(x), y, label_smoothing=rc["label_smoothing"])
                cost = g.soft_cost()
                delta = (cost - budget) / budget
                loss = task.float() + rc["alpha"] * g.sens_reg()
                if apply_budget:
                    loss = loss + lam * delta + 0.5 * rho * delta ** 2
                scaler.scale(loss).backward()
                if rc["clip_a"] > 0:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_([p for grp in opt.param_groups for p in grp["params"]],
                                                   rc["clip_a"])
                scaler.step(opt)
                scaler.update()
                step += 1
                sums["task"] += float(task.detach())
                sums["cost"] += float(cost.detach())
                sums["n"] += 1
                if apply_budget:
                    deltas.append(float(delta.detach()))
            if apply_budget and deltas:
                lam = max(0.0, min(rc["lambda_max"], lam + rc["eta"] * sum(deltas) / len(deltas)))
            if sched:
                sched.step()
            hard = g.hard_cost()
            kept = g.hard_counts()
            rec = {"phase": "A", "epoch": e + 1, "budget_active": apply_budget,
                   "train_task_loss": sums["task"] / max(sums["n"], 1),
                   "soft_cost_macs": sums["cost"] / max(sums["n"], 1), "hard_cost_macs": hard,
                   "hard_cost_fraction": hard / full, "lambda": lam, "beta": g.beta, "rho": rho,
                   "kept_units": kept, "seconds": time.time() - t0}
            lg["history"].append(rec)
            log("[A %d/%d]%s loss %.4f  soft %.1fM  hard %.1fM (%.1f%% of unpruned, budget %.0f%%)  lambda %.3f  (%.0fs)"
                % (e + 1, rc["epochs_a"], "" if apply_budget else " warm-up", rec["train_task_loss"],
                   rec["soft_cost_macs"] / 1e6, hard / 1e6, 100 * hard / full, 100 * rc["budget"], lam,
                   rec["seconds"]))
            run.save({"phase": "A", "epoch": e + 1, "gated_sd": g.state_dict(), "opt": opt.state_dict(),
                      "sched": sched.state_dict() if sched else None, "scaler": scaler.state_dict(),
                      "lam": lam, "step": step, "log": lg})

        g.eval()
        before = g.hard_cost()
        kept_before = g.hard_counts()
        fit = g.fit_budget(budget if not a.no_enforce_budget else float("inf"), a.round_to)
        closed = fit["units_closed"]
        after = g.hard_cost()
        torch.save(g.state_dict()["gates"], os.path.join(run.out, "gates.pt"))
        pruned = export_pruned(g)
        xb = torch.stack([eval_loader.dataset[i][0] for i in range(min(8, len(eval_loader.dataset)))])
        g_cpu = g.to("cpu")
        diff = compare_outputs(g_cpu, pruned, xb)
        measured = eb.count_macs(pruned, a.input_size)["macs"]
        g.remove_hooks()
        lg["export"] = {"hard_cost_before_projection": before, "kept_units_before_projection": kept_before,
                        "projection": fit, "units_closed_by_budget_projection": closed,
                        "cost_model_macs": after, "measured_macs": measured,
                        "cost_model_equals_measured": after == measured,
                        "gated_vs_exported_max_abs_logit_diff": diff, "kept_units": g.hard_counts(),
                        "budget_met": measured <= budget + 0.5}
        lg["pruned_before_finetune"] = complexity(pruned, a.input_size)
        log("export: %s params, %.2f M MACs (%.1f%% of unpruned; %d units closed to meet budget); "
            "gated vs exported max |diff| %.2e" % (f"{lg['pruned_before_finetune']['params']:,}", measured / 1e6,
                                                     100 * measured / full, closed, diff))
        log("        (binary gates before rounding/projection: %.2f M MACs = %.1f%%; %d units opened by rounding "
            "to multiples of %d)" % (before / 1e6, 100 * before / full, fit["units_opened"], a.round_to))
        if after != measured:
            log("WARNING: cost model %d != measured %d" % (after, measured))
        if diff > 1e-3:
            log("WARNING: exported model differs from the gated model (max |diff| %.2e)" % diff)
        lg["export"]["checkpoint"] = save_model(pruned, a.arch, os.path.join(run.out, "exported_before_finetune.pt"))
        del g, g_cpu
        pruned.to(dev)
        pre = evaluate(pruned, eval_loader, dev, a.amp)
        lg["eval_before_finetune"] = dict(pre, split=info["eval_split"],
                                          note="reported only; not used for any decision")
        log("before fine-tune %s top-1: %.2f%%" % (info["eval_split"], 100 * pre["top1"]))
        state = None

    phase_b(pruned, a, rc, dev, train_set, run, state, lg, a.arch)
    finish(a, run, pruned, a.arch, lg, eval_loader, dev, t_start)



def _depgraph_ignored(model):
    """Every layer whose output channels are coupled to an SDTA-stage trunk or
    to XCA heads: Torch-Pruning cannot index those groups on EdgeNeXt (the
    IndexError), so they are kept intact. Prunable: MLP hidden units of every
    block and the stage-1 trunk."""
    ig = [model.head.fc]
    for s, st in enumerate(model.stages):
        if s >= 1 and len(st.downsample) > 1:
            ig.append(st.downsample[1])
        for b in st.blocks:
            if s >= 1:
                ig.append(b.mlp.fc2)
            if hasattr(b, "xca"):
                ig += [b.xca.qkv, b.xca.proj] + list(b.convs)
                if b.pos_embd is not None:
                    ig.append(b.pos_embd.token_projection)
    return ig


def depgraph_prune(model, ratio, input_size, round_to=None):
    import torch
    import torch.nn as nn
    import torch_pruning as tp
    m = copy.deepcopy(model).cpu().eval()
    ex = torch.randn(1, 3, input_size, input_size)
    imp_cls = getattr(tp.importance, "GroupMagnitudeImportance", None) or tp.importance.MagnitudeImportance
    kw = dict(importance=imp_cls(p=2), ignored_layers=_depgraph_ignored(m), root_module_types=[nn.Conv2d, nn.Linear])
    if round_to and round_to > 1:
        kw["round_to"] = round_to
    try:
        pruner = tp.pruner.MetaPruner(m, ex, pruning_ratio=ratio, **kw)
    except TypeError:
        pruner = tp.pruner.MetaPruner(m, ex, ch_sparsity=ratio, **kw)
    pruner.step()
    return m


def cmd_depgraph(a):
    import warnings
    t_start = time.time()
    try:
        import torch_pruning as tp
    except ImportError:
        raise SystemExit("pip install torch-pruning")
    rc, dev, train_set, eval_loader, info = setup(a, "depgraph")
    run = Run(a.out)
    state = resume_state(a, run)
    if state:
        model, _ = eb.rebuild_from_state_dict(state["model_sd"], a.arch)
        lg = state["log"]
    else:
        model = new_model(a, info["num_classes"])
        lg = common_log(a, rc, info, dev, "depgraph")
        lg["unpruned"] = complexity(model, a.input_size)
    model.to(dev)
    if not state or state["phase"] == "A":
        dense_phase_a(model, a, rc, dev, train_set, run, state, lg, a.arch)
        state = None
        full = lg["unpruned"]["macs"]
        target = rc["budget"] * full
        warnings.filterwarnings("ignore", message="Unwrapped parameters detected")
        grid = [round(x, 4) for x in np.arange(0.05, a.max_ratio + 1e-9, 0.05)]
        tried = []
        for r in grid:
            cand = depgraph_prune(model, r, a.input_size, a.round_to)
            tried.append((r, eb.count_macs(cand, a.input_size)["macs"], cand))
            if tried[-1][1] <= target:
                break
        feasible = [t for t in tried if t[1] <= target]
        if feasible:
            reached = True
            hi, best = feasible[0][0], feasible[0][2]
            lo = tried[-2][0] if len(tried) > 1 else 0.0
            for _ in range(a.bisect_steps):
                mid = (lo + hi) / 2
                cand = depgraph_prune(model, mid, a.input_size, a.round_to)
                if eb.count_macs(cand, a.input_size)["macs"] <= target:
                    hi, best = mid, cand
                else:
                    lo = mid
        else:
            reached = False
            hi, _, best = min(tried, key=lambda t: t[1])
        model = best
        macs = eb.count_macs(model, a.input_size)["macs"]
        lg["budget"] = {"fraction": rc["budget"], "unpruned_macs": full, "target_macs": target}
        lg["depgraph"] = {"importance": "GroupMagnitudeImportance(p=2)", "round_to": a.round_to,
                          "ratio": hi,
                          "budget_reachable": reached, "macs": macs,
                          "pruned_groups": "MLP hidden of every block + stage-1 trunk",
                          "kept_intact": "SDTA-stage trunks (stages 2-4), XCA heads, classifier"}
        lg["pruned_before_finetune"] = complexity(model, a.input_size)
        log("DepGraph: ratio %.3f -> %.2f M MACs (%.1f%% of unpruned)%s"
            % (lg["depgraph"]["ratio"], macs / 1e6, 100 * macs / full,
               "" if reached else "  BUDGET NOT REACHABLE with the prunable groups"))
        model.to(dev)
        pre = evaluate(model, eval_loader, dev, a.amp)
        lg["eval_before_finetune"] = dict(pre, split=info["eval_split"], note="reported only")
        log("before fine-tune %s top-1: %.2f%%" % (info["eval_split"], 100 * pre["top1"]))
    phase_b(model, a, rc, dev, train_set, run, state, lg, a.arch)
    finish(a, run, model, a.arch, lg, eval_loader, dev, t_start)



def _randomize(model, seed):
    """Non-trivial weights: LayerScale 1 instead of 1e-6, random biases and
    norm affine parameters, so a wrong index cannot hide."""
    import torch
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith("gamma") or name.endswith("gamma_xca"):
                p.copy_(1.0 + 0.1 * torch.randn(p.shape, generator=gen))
            elif name.endswith("bias"):
                p.copy_(0.1 * torch.randn(p.shape, generator=gen))
            elif ("norm" in name or "stem.1" in name or "downsample.0" in name) and name.endswith("weight"):
                p.copy_(1.0 + 0.1 * torch.randn(p.shape, generator=gen))
            elif name.endswith("temperature"):
                p.copy_(0.5 + torch.rand(p.shape, generator=gen))


def cmd_check(a):
    import timm
    import torch
    eb.install_edgenext_patches()
    ok = True
    rows = []

    def record(name, value, passed):
        nonlocal ok
        ok = ok and passed
        rows.append((name, value, "PASS" if passed else "FAIL"))
        log("  %-62s %-14s %s" % (name, value, "PASS" if passed else "FAIL"))

    archs = a.archs.split(",")
    for arch in archs:
        for gate_types in (",".join(GATE_TYPES), "trunk,heads,mlp"):
            log("\n%s, gates: %s" % (arch, gate_types))
            torch.manual_seed(a.seed)
            base = timm.create_model(arch, pretrained=False, num_classes=a.num_classes)
            _randomize(base, a.seed)
            ref = copy.deepcopy(base).eval()
            g = GatedEdgeNeXt(base, arch, a.num_classes, a.input_size, gate_types.split(","))
            x = torch.randn(2, 3, a.input_size, a.input_size, generator=torch.Generator().manual_seed(1))
            g.eval()
            record("all gates open: gated == original (masked LayerNorm)", "%.1e" % compare_outputs(g, ref, x),
                   compare_outputs(g, ref, x) < 1e-4)
            measured = eb.count_macs(ref, a.input_size)["macs"]
            record("cost model (all open) == measured MACs", "%d" % g.full_cost(), g.full_cost() == measured)
            gen = torch.Generator().manual_seed(a.seed + 7)
            with torch.no_grad():
                for k, la in g.gates.items():
                    la.copy_(torch.randn(la.shape, generator=gen) + 0.4)
            pruned = export_pruned(g)
            d = compare_outputs(g, pruned, x)
            record("random gates: gated (eval) == exported", "%.1e" % d, d < 1e-4)
            mm = eb.count_macs(pruned, a.input_size)["macs"]
            record("cost model (binary gates) == measured MACs of exported", "%.2f%% of full" % (100 * mm / measured),
                   g.hard_cost() == mm)
            budget = 0.5 * measured
            fit = g.fit_budget(budget, 8)
            pruned2 = export_pruned(g)
            mm2 = eb.count_macs(pruned2, a.input_size)["macs"]
            record("rounding to x8 + projection: exported MACs <= 50% budget",
                   "%.2f%% (-%d/+%d)" % (100 * mm2 / measured, fit["units_closed"], fit["units_opened"]),
                   mm2 <= budget and g.hard_cost() == mm2)
            hc = g.hard_counts()
            aligned = all(v % 8 == 0 or v == g.gates[k].numel() for k, v in hc.items() if not k.startswith("head"))
            record("kept counts are multiples of 8 (or full)", "", aligned)
            d2 = compare_outputs(g, pruned2, x)
            record("after projection: gated (eval) == exported", "%.1e" % d2, d2 < 1e-4)
            rb, arch_inf = eb.rebuild_from_state_dict(pruned2.state_dict())
            d3 = compare_outputs(rb, pruned2, x)
            record("rebuild from state_dict == exported (arch inferred: %s)" % arch_inf, "%.1e" % d3,
                   d3 < 1e-6 and arch_inf == arch)
            summary = eb.arch_summary(pruned2)
            log("    exported: dims %s, heads %s, sdta split %s" % (summary["stage_dims"], summary["heads"],
                                                                   summary["sdta_split"]))
            g.train()
            g.beta = 0.5
            out = g(x)
            loss = out.float().logsumexp(1).mean() + 1e-9 * g.soft_cost()
            loss.backward()
            has_grad = all(p.grad is not None and torch.isfinite(p.grad).all() for p in g.parameters_gates())
            record("training step: finite gradients on every gate", "", bool(has_grad))
            record("Taylor sensitivities collected for every MLP gate", "%d" % len(g.sens),
                   len(g.sens) == sum(1 for k in g.gates if k.startswith("mlp")))
            g.remove_hooks()
            if a.onnx or a.tflite:
                import tempfile
                with tempfile.TemporaryDirectory() as td:
                    x1 = x[:1].numpy()
                    with torch.no_grad():
                        y1 = pruned2(x[:1]).numpy()
                    if a.onnx:
                        p = os.path.join(td, "m.onnx")
                        eb.export_onnx(pruned2, p, a.input_size, 17)
                        r = eb.OrtRunner(p, 1)
                        do = float(np.abs(eb.run_logits(r, x1) - y1).max())
                        record("ONNX Runtime == PyTorch (exported model)", "%.1e" % do, do < 1e-3)
                    if a.tflite:
                        p = os.path.join(td, "m.tflite")
                        eb.export_tflite_fp32(pruned2, p, a.input_size)
                        r = eb.TfliteRunner(p, 1)
                        dt = float(np.abs(eb.run_logits(r, x1) - y1).max())
                        record("TFLite FP32 == PyTorch (exported model)", "%.1e" % dt, dt < 1e-3)
    log("\n%s" % ("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED"))
    if not ok:
        sys.exit(1)



def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp, budget):
        sp.add_argument("--arch", required=True, choices=sorted(set(eb.ARCH_BY_STAGE4_DIM.values())))
        sp.add_argument("--dataset", required=True, choices=list(eb.DATASETS))
        sp.add_argument("--data-root", required=True)
        sp.add_argument("--out", required=True, help="output folder (model.pt, log.json)")
        sp.add_argument("--recipe", choices=list(RECIPES),
                        help="default: xxs for edgenext_xx_small, s otherwise")
        sp.add_argument("--seed", type=int, default=0)
        sp.add_argument("--epochs-a", type=int, help="phase A epochs (recipe default)")
        sp.add_argument("--epochs-b", type=int, help="phase B epochs (recipe default)")
        sp.add_argument("--batch-size", type=int)
        sp.add_argument("--input-size", type=int, default=224)
        sp.add_argument("--class-order", help="tiny10/tiny200: comma-separated WNIDs in label order")
        sp.add_argument("--train-per-class", type=int, help="cap train images per class (smoke tests only)")
        sp.add_argument("--eval-per-class", type=int, help="cap eval images per class (smoke tests only)")
        sp.add_argument("--workers", type=int, default=2)
        sp.add_argument("--amp", action="store_true", help="mixed precision on CUDA (use for all methods or none)")
        sp.add_argument("--cpu", action="store_true")
        sp.add_argument("--no-pretrained", action="store_true")
        sp.add_argument("--download", action="store_true", help="CIFAR only")
        sp.add_argument("--resume", action="store_true")
        sp.add_argument("--keep-resume", action="store_true", help="keep resume.pt after the run")
        sp.add_argument("--override", nargs="*", metavar="KEY=VALUE", help="override recipe values")
        if budget:
            sp.add_argument("--budget", type=float, help="target MACs as a fraction of the unpruned model")

    b = sub.add_parser("baseline", help="unpruned model, same recipe")
    common(b, False)
    b.set_defaults(func=cmd_baseline)

    g = sub.add_parser("gated", help="proposed coupling-typed gated pruning")
    common(g, True)
    g.add_argument("--gate-types", default=",".join(GATE_TYPES),
                   help="comma list from %s; 'trunk,heads,mlp' = archived behaviour "
                        "(SDTA-stage trunks kept full)" % ",".join(GATE_TYPES))
    g.add_argument("--warmup-epochs", type=int, help="epochs without budget loss (recipe default)")
    g.add_argument("--init-logit", type=float, default=2.0)
    g.add_argument("--sens-ema", type=float, default=0.9)
    g.add_argument("--no-enforce-budget", action="store_true",
                   help="do not close extra gates if the binary gates exceed the budget")
    g.add_argument("--round-to", type=int, default=8,
                   help="kept channel / hidden-unit counts are multiples of this (1 = off). Unaligned "
                        "widths can erase the latency gain on CPU runtimes")
    g.set_defaults(func=cmd_gated)

    d = sub.add_parser("depgraph", help="Torch-Pruning baseline on the groups it can handle")
    common(d, True)
    d.add_argument("--max-ratio", type=float, default=0.95)
    d.add_argument("--round-to", type=int, default=8, help="Torch-Pruning round_to (1 = off)")
    d.add_argument("--bisect-steps", type=int, default=8)
    d.set_defaults(func=cmd_depgraph)

    c = sub.add_parser("check", help="self-test, no data needed")
    c.add_argument("--archs", default="edgenext_xx_small,edgenext_small")
    c.add_argument("--num-classes", type=int, default=10)
    c.add_argument("--input-size", type=int, default=224)
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--onnx", action="store_true", help="also check ONNX Runtime parity")
    c.add_argument("--tflite", action="store_true", help="also check TFLite FP32 parity")
    c.set_defaults(func=cmd_check)

    args = p.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
