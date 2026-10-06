# file:        exp_ml_ntc_v3.py
# description: learned codec definitions used by the architecture report
# author:      Mitchell A. Thornton <mitch@smu.edu>, ORCID 0000-0003-3559-9511
# license:     MIT (see LICENSE)
"""Learned nonlinear transform coding baseline (Balle-style), trained on the
v3 (2026-10-02) changes from v2: (1) --device auto picks CUDA when present (the
Spark GB10), then Apple MPS, then CPU, and on CUDA the training frames are moved to
the GPU once; the random batch indices are drawn on the CPU from the same seeded
generator as in v2, so a run is reproducible on one device; (2) --extra-test NPZ
scores every model on a second frame set after training, for the distribution-shift
test (train on CSPB.ML.2018R2, score on the blind CSPB.ML.2022R2 set); with
--extra-operational JSON the AD operational rates of that set are paired signal by
signal; (3) accepts the v2 frame sets; result names carry the v3 prefix.
training groups of the ML dataset and scored on the same held-out frames as
the AD modes and PCA, at matched distortion, on an operational rate.

Purpose: the nonlinear learned codec the AD transform coder must be compared
with. The construction is the standard one of end-to-end optimized
compression (Balle, Laparra and Simoncelli; Balle et al.): a convolutional
analysis transform, additive uniform noise in training and rounding at test,
a per-channel factorized nonparametric entropy model (the cumulative-logit
network of Balle et al., as implemented in CompressAI's EntropyBottleneck),
a convolutional synthesis transform, and the loss rate + lambda x distortion.
It is written in plain PyTorch so it needs no compiled extension.
Created in bundle v9.5. Last modified in bundle v9.5.
(C) Mitchell A. Thornton 2026

Script history: v1 2026-09-30, first version. v2 2026-09-30: adds the mean-scale
hyperprior model (Balle et al. 2018; Minnen, Balle and Toderici 2018), now the
default, in which a second, coded latent sets a Gaussian mean and scale for
every first-level latent, so the entropy model adapts to each frame and pays
for that adaptation in bits (v1's factorized model is one fixed density for all
signals, while the AD and PCA modes adapt their coefficient variances to each
signal); logs the training rate and distortion every --log-every iterations,
so convergence can be read from the console; raises the default iteration
count from 20,000 to 60,000 (v1's rate was still falling between 3,000 and
20,000 iterations). Written without the ability to run PyTorch where it was
written: run --smoke first and check its PASS lines.

Protocol:
  input     frames of M complex samples as 2 x M real (I, Q); --input der uses
            the derotated frames (the same preprocessing the AD widely linear
            modes use, and those rows pay the same carrier side information),
            --input raw the channelized frames without carrier removal
  training  frames of the training groups only (both halves), optionally one
            class (--train-class) and a limited number of signals
            (--train-signals) for the sample-efficiency curve
  lambdas   one model per lambda; each held-out signal gets (rate, distortion)
            per lambda on its second-half frames
  rate      ideal code length -log2 p of the rounded latents under the learned
            entropy model, bits per complex sample (a range coder is within a
            fraction of a percent of this); the model is fixed at both ends, so
            there is no per-signal side information
  matching  per signal, the rate at distortion RHO times the mean frame energy,
            by linear interpolation in log distortion between the two lambdas
            that bracket it; a signal not bracketed is reported as unmatched,
            never extrapolated
  compare   with --operational results/ml_operational_v1_<tag>.json the script
            prints the AD and PCA operational rates of the same signals beside
            the learned codec

Run (Mac, adml environment):
  conda activate adml
  python scripts/exp_ml_ntc_v3.py --dataset results/ml_dataset_v2_cspb2018r2_groups1000_seed0.npz --smoke
  python scripts/exp_ml_ntc_v3.py --dataset results/ml_dataset_v2_cspb2018r2_groups1000_seed0.npz \
      --operational results/ml_operational_v1_groups1000_seed0.json
Writes results/ml_ntc_v2_<dataset tag>_<model>_<input>_<train class>_<train signals>.json.
"""
import re
import argparse, json, math, os, sys, time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fnn

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SCRIPT = "exp_ml_ntc_v3"


# ------------------------------------------------------------ entropy model
class FactorizedDensity(nn.Module):
    """Per-channel univariate density through a monotone cumulative-logit
    network (Balle et al. 2018, appendix; CompressAI EntropyBottleneck)."""

    def __init__(self, C, filters=(3, 3, 3), init_scale=10.0):
        super().__init__()
        f = (1,) + tuple(filters) + (1,)
        scale = init_scale ** (1.0 / (len(filters) + 1))
        self.mats, self.biases, self.factors = (nn.ParameterList(), nn.ParameterList(),
                                                nn.ParameterList())
        for i in range(len(filters) + 1):
            init = math.log(math.expm1(1.0 / scale / f[i + 1]))
            self.mats.append(nn.Parameter(torch.full((C, f[i + 1], f[i]), init)))
            self.biases.append(nn.Parameter(torch.empty(C, f[i + 1], 1).uniform_(-0.5, 0.5)))
            if i < len(filters):
                self.factors.append(nn.Parameter(torch.zeros(C, f[i + 1], 1)))

    def _logits(self, v):                       # v: (C, 1, N)
        h = v
        for i in range(len(self.mats)):
            h = torch.matmul(Fnn.softplus(self.mats[i]), h) + self.biases[i]
            if i < len(self.factors):
                h = h + torch.tanh(self.factors[i]) * torch.tanh(h)
        return h

    def likelihood(self, y):                    # y: (B, C, L)
        B, C, L = y.shape
        v = y.permute(1, 0, 2).reshape(C, 1, -1)
        lo, hi = self._logits(v - 0.5), self._logits(v + 0.5)
        sgn = -torch.sign(lo + hi).detach()
        p = torch.abs(torch.sigmoid(sgn * hi) - torch.sigmoid(sgn * lo))
        return p.clamp_min(1e-9).reshape(C, B, L).permute(1, 0, 2)


# ------------------------------------------------------------------- codec
class NTC(nn.Module):
    def __init__(self, C=32, hidden=128):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv1d(2, hidden, 5, padding=2), nn.GELU(),
            nn.Conv1d(hidden, hidden, 5, stride=2, padding=2), nn.GELU(),
            nn.Conv1d(hidden, hidden, 5, stride=2, padding=2), nn.GELU(),
            nn.Conv1d(hidden, C, 5, padding=2))
        self.dec = nn.Sequential(
            nn.Conv1d(C, hidden, 5, padding=2), nn.GELU(),
            nn.ConvTranspose1d(hidden, hidden, 4, stride=2, padding=1), nn.GELU(),
            nn.ConvTranspose1d(hidden, hidden, 4, stride=2, padding=1), nn.GELU(),
            nn.Conv1d(hidden, 2, 5, padding=2))
        self.density = FactorizedDensity(C)

    def forward(self, x, train=True):
        y = self.enc(x)
        yq = y + torch.empty_like(y).uniform_(-0.5, 0.5) if train else torch.round(y)
        return self.dec(yq), self.density.likelihood(yq)


SCALE_LB = 0.11


def gaussian_likelihood(yq, mu, sigma):
    """Probability of the unit bin around yq under N(mu, sigma^2)."""
    c = 1.0 / math.sqrt(2.0)
    v = torch.abs(yq - mu)
    up = 0.5 * torch.erfc(-(0.5 - v) / sigma * c)
    lo = 0.5 * torch.erfc(-(-0.5 - v) / sigma * c)
    return (up - lo).clamp_min(1e-9)


class NTCHyper(nn.Module):
    """Mean-scale hyperprior: y = g_a(x); z = h_a(y) coded with a factorized
    density; (mu, sigma) = h_s(z_hat) give the Gaussian conditional of y."""

    def __init__(self, C=32, Cz=16, hidden=128, hyper=64):
        super().__init__()
        base = NTC(C, hidden)
        self.enc, self.dec = base.enc, base.dec
        self.h_a = nn.Sequential(
            nn.Conv1d(C, hyper, 3, padding=1), nn.GELU(),
            nn.Conv1d(hyper, hyper, 5, stride=2, padding=2), nn.GELU(),
            nn.Conv1d(hyper, Cz, 3, padding=1))
        self.h_s = nn.Sequential(
            nn.ConvTranspose1d(Cz, hyper, 4, stride=2, padding=1), nn.GELU(),
            nn.Conv1d(hyper, hyper, 3, padding=1), nn.GELU(),
            nn.Conv1d(hyper, 2 * C, 3, padding=1))
        self.density = FactorizedDensity(Cz)

    def forward(self, x, train=True):
        y = self.enc(x)
        z = self.h_a(y)
        zq = z + torch.empty_like(z).uniform_(-0.5, 0.5) if train else torch.round(z)
        pz = self.density.likelihood(zq)
        mu, s = self.h_s(zq).chunk(2, dim=1)
        sigma = Fnn.softplus(s) + SCALE_LB
        if train:
            yq = y + torch.empty_like(y).uniform_(-0.5, 0.5)
        else:
            yq = torch.round(y - mu) + mu
        py = gaussian_likelihood(yq, mu, sigma)
        return self.dec(yq), (py, pz)


def total_bits(p):
    if isinstance(p, tuple):
        return sum(-torch.log2(q).sum() for q in p)
    return -torch.log2(p).sum()


def to_tensor(frames):                          # complex (n, M) -> (n, 2, M) float32
    return torch.from_numpy(np.stack([frames.real, frames.imag], 1).astype(np.float32))


def train_one(X, lam, iters, batch, device, seed, model="hyperprior", log_every=0):
    torch.manual_seed(seed)
    net = (NTCHyper() if model == "hyperprior" else NTC()).to(device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, iters)
    M = X.shape[2]
    n = X.shape[0]
    g = torch.Generator().manual_seed(seed)
    Xd = X.to(device) if str(device).startswith("cuda") else X
    for it in range(iters):
        idx = torch.randint(0, n, (batch,), generator=g)
        x = Xd[idx.to(Xd.device)] if Xd is not X else X[idx].to(device)
        xh, p = net(x, train=True)
        rate = total_bits(p) / (batch * M)
        dist = ((x - xh) ** 2).sum() / (batch * M)
        loss = rate + lam * dist
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if log_every:
            if it % log_every == 0:
                acc_r, acc_d, acc_n = rate.detach() * 0, dist.detach() * 0, 0
            acc_r = acc_r + rate.detach(); acc_d = acc_d + dist.detach(); acc_n += 1
            if (it + 1) % log_every == 0:          # the only device sync in the loop
                print(f"      iter {it + 1:6d}: train rate {float(acc_r) / acc_n:.3f}, "
                      f"distortion {float(acc_d) / acc_n:.4f}")
    return net


@torch.no_grad()
def evaluate(net, frames, device):
    x = to_tensor(frames).to(device)
    xh, p = net(x, train=False)
    M = x.shape[2]
    bits = float(total_bits(p)) / (x.shape[0] * M)
    dist = float(((x - xh) ** 2).sum()) / (x.shape[0] * M)
    return bits, dist


def rate_at(points, D):
    """points: list of (rate, dist); linear in log dist between the bracketing pair."""
    pts = sorted(points, key=lambda t: t[1])
    for (r1, d1), (r2, d2) in zip(pts[:-1], pts[1:]):
        if d1 <= D <= d2 and d1 > 0:
            w = (math.log(D) - math.log(d1)) / (math.log(d2) - math.log(d1)) if d2 > d1 else 0.0
            return r1 + w * (r2 - r1)
    return None


@torch.no_grad()
def smoke_checks(net, device):
    C = net.density.mats[0].shape[0]
    k = torch.arange(-60, 61, dtype=torch.float32, device=device)
    y = k.view(1, 1, -1).expand(1, C, -1).contiguous()
    tot = net.density.likelihood(y).sum(dim=2).squeeze(0)
    ok = bool(torch.all((tot > 0.98) & (tot < 1.02)))
    print(f"  entropy model sums to one over integers (per channel, range {float(tot.min()):.4f}"
          f" to {float(tot.max()):.4f}): {'PASS' if ok else 'FAIL'}")
    return ok


@torch.no_grad()
def gaussian_check(device):
    k = torch.arange(-200, 201, dtype=torch.float32, device=device).view(1, 1, -1)
    mu = torch.tensor(0.3, device=device); sg = torch.tensor(3.7, device=device)
    tot = float(gaussian_likelihood(k, mu, sg).sum())
    ok = abs(tot - 1.0) < 1e-3
    print(f"  Gaussian conditional sums to one over integers ({tot:.5f}): {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--operational", default=None)
    ap.add_argument("--input", choices=["der", "raw"], default="der")
    ap.add_argument("--train-class", default="all")
    ap.add_argument("--train-signals", type=int, default=0, help="0 = all")
    ap.add_argument("--lambdas", default="30,60,120,240,480,960")
    ap.add_argument("--iters", type=int, default=60000)
    ap.add_argument("--model", choices=["hyperprior", "factorized"], default="hyperprior")
    ap.add_argument("--log-every", type=int, default=5000)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--rho", type=float, default=0.01)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--extra-test", default=None, help="second frame set scored after training")
    ap.add_argument("--extra-operational", default=None,
                    help="AD operational rates of the second set (exp_ml_operational_v3.py)")
    a = ap.parse_args()
    t0 = time.time()
    if a.device == "auto":
        dev = ("cuda" if torch.cuda.is_available() else
               "mps" if torch.backends.mps.is_available() else "cpu")
    else:
        dev = a.device
    if a.smoke:
        a.iters, lambdas = 300, [60.0, 480.0]
    else:
        lambdas = [float(x) for x in a.lambdas.split(",")]
    d = np.load(a.dataset, allow_pickle=False)
    A = d["der"] if a.input == "der" else d["raw"]
    mods, held = d["sig_mod"], d["sig_heldout"].astype(bool)
    st, nfr, half, cb = d["sig_start"], d["sig_nfr"], d["sig_half"], d["sig_carrier_bps"]
    fr_of = lambda i: A[st[i]:st[i] + nfr[i]].astype(np.complex64)
    tr = [i for i in np.where(~held)[0] if a.train_class in ("all", str(mods[i]))]
    if a.train_signals:
        tr = tr[:a.train_signals]
    te = list(np.where(held)[0])
    if a.smoke:
        tr, te = tr[:40], te[:24]
    X = to_tensor(np.concatenate([fr_of(i) for i in tr]))
    M = X.shape[2]
    print(f"{SCRIPT}: model {a.model}; torch {torch.__version__} on {dev}; input {a.input}; training class "
          f"{a.train_class}, {len(tr)} signals, {X.shape[0]} frames; {len(te)} held-out signals; "
          f"lambdas {lambdas}; {a.iters} iterations each")
    per = {int(i): [] for i in te}
    E = None
    if a.extra_test:
        e = np.load(a.extra_test, allow_pickle=False)
        EA = e["der"] if a.input == "der" else e["raw"]
        e_te = list(np.where(e["sig_heldout"].astype(bool))[0])
        if a.smoke:
            e_te = e_te[:24]
        E = dict(d=e, A=EA, te=e_te, per={int(i): [] for i in e_te})
        e_fr = lambda i: EA[e["sig_start"][i]:e["sig_start"][i] + e["sig_nfr"][i]].astype(np.complex64)
        print(f"extra test set {os.path.basename(a.extra_test)}: {len(e_te)} signals")
    for j, lam in enumerate(lambdas):
        net = train_one(X, lam, a.iters, a.batch, dev, a.seed + j, a.model,
                        100 if a.smoke else a.log_every)
        net.eval()
        if a.smoke:
            smoke_checks(net, dev)
            if a.model == "hyperprior" and j == 0:
                gaussian_check(dev)
        R = []
        for i in te:
            fr = fr_of(i)[int(half[i]):]
            b, dd = evaluate(net, fr, dev)
            per[int(i)].append((b, dd))
            R.append((b, dd))
        if E is not None:
            for i in E["te"]:
                E["per"][int(i)].append(evaluate(net, e_fr(i)[int(E["d"]["sig_half"][i]):], dev))
        R = np.array(R)
        print(f"  lambda {lam:7.1f}: median rate {np.median(R[:, 0]):.3f} bits/sample, median "
              f"distortion {np.median(R[:, 1]):.4f} (target {a.rho:.4f} x energy) "
              f"[{time.time() - t0:.0f} s]")
    rows, unmatched = [], 0
    for i in te:
        fr = fr_of(i)[int(half[i]):]
        D = a.rho * float(np.mean(np.sum(np.abs(fr) ** 2, axis=1))) / M
        r = rate_at(per[int(i)], D)
        if r is not None and a.input == "der":
            r += float(cb[i])
        unmatched += r is None
        rows.append(dict(index=int(d["sig_index"][i]), mod=str(mods[i]),
                         snr=float(d["sig_snr"][i]), target_dist=D, ntc=r,
                         points=per[int(i)]))
    print(f"\nunmatched (target distortion outside the lambda range): {unmatched} of {len(rows)}")
    if a.smoke:
        ordered = all(p[0][0] < p[-1][0] for p in per.values())
        print(f"  higher lambda gives higher rate for every held-out signal: "
              f"{'PASS' if ordered else 'FAIL'}")
        finite = all(np.isfinite(np.array(p)).all() for p in per.values())
        print(f"  all rates and distortions finite: {'PASS' if finite else 'FAIL'}")
    op = None
    if a.operational:
        op = {s["index"]: s for s in json.load(open(a.operational))["signals"]}
    print("\nmedian operational rate over matched held-out signals, bits per complex sample")
    hdr = f"{'class':>7} {'n':>4} {'NTC':>8}" + (f" {'AD':>8} {'PCA-p':>8} {'PCA-WL':>8} {'NTC-AD':>8}"
                                                  if op else "")
    print(hdr)
    summ = {}
    for m in sorted(set(r["mod"] for r in rows), key=lambda z: z):
        R = [r for r in rows if r["mod"] == m and r["ntc"] is not None]
        if not R:
            continue
        v = dict(n=len(R), ntc=float(np.median([r["ntc"] for r in R])))
        line = f"{m:>7} {len(R):4d} {v['ntc']:8.3f}"
        if op:
            Q = [r for r in R if r["index"] in op]
            v["ad"] = float(np.median([op[r["index"]]["ad"] for r in Q]))
            v["pca_p"] = float(np.median([op[r["index"]]["pca_p_cls"] for r in Q]))
            v["pca_w"] = float(np.median([op[r["index"]]["pca_w_cls"] for r in Q]))
            v["ntc_minus_ad"] = float(np.median([r["ntc"] - op[r["index"]]["ad"] for r in Q]))
            line += f" {v['ad']:8.3f} {v['pca_p']:8.3f} {v['pca_w']:8.3f} {v['ntc_minus_ad']:+8.3f}"
        summ[m] = v
        print(line)
    extra = None
    if E is not None:
        e = E["d"]; eop = None
        if a.extra_operational:
            eop = {s_["index"]: s_ for s_ in json.load(open(a.extra_operational))["signals"]}
        erows, eun = [], 0
        for i in E["te"]:
            fr = e_fr(i)[int(e["sig_half"][i]):]
            D = a.rho * float(np.mean(np.sum(np.abs(fr) ** 2, axis=1))) / M
            r = rate_at(E["per"][int(i)], D)
            if r is not None and a.input == "der":
                r += float(e["sig_carrier_bps"][i])
            eun += r is None
            idx_ = int(e["sig_index"][i])
            row = dict(index=idx_, target_dist=D, ntc=r, points=E["per"][int(i)])
            if eop and idx_ in eop:
                row.update(ad=eop[idx_]["ad"], ad_selected=eop[idx_]["ad_selected"])
            erows.append(row)
        M_ = [r for r in erows if r["ntc"] is not None]
        es = dict(n=len(erows), unmatched=eun, ntc=(float(np.median([r["ntc"] for r in M_])) if M_ else None))
        P = [r for r in M_ if "ad" in r]
        if P:
            es.update(n_paired=len(P), ad=float(np.median([r["ad"] for r in P])),
                      ntc_minus_ad=float(np.median([r["ntc"] - r["ad"] for r in P])),
                      frac_ad_shorter=float(np.mean([r["ntc"] > r["ad"] for r in P])))
            for fam, f in (("wl", lambda r: r["ad_selected"].startswith("WL")),
                           ("proper", lambda r: not r["ad_selected"].startswith("WL"))):
                Q = [r for r in P if f(r)]
                if Q:
                    es[fam] = dict(n=len(Q), ntc=float(np.median([r["ntc"] for r in Q])),
                                   ad=float(np.median([r["ad"] for r in Q])),
                                   ntc_minus_ad=float(np.median([r["ntc"] - r["ad"] for r in Q])))
        extra = dict(dataset=os.path.basename(a.extra_test), summary=es, signals=erows)
        print(f"\nextra test set: {es}")
    tag = re.sub(r"^ml_dataset_v[12]_", "", os.path.splitext(os.path.basename(a.dataset))[0])
    tag += f"_{a.model}_{a.input}_{a.train_class}_{a.train_signals or 'allsig'}" + ("_smoke" if a.smoke else "")
    if a.extra_test:
        tag += "_x" + re.sub(r"^ml_dataset_v[12]_", "", os.path.splitext(os.path.basename(a.extra_test))[0]).split("_")[0]
    fn = os.path.join(ROOT, "results", f"ml_ntc_v3_{tag}.json")
    with open(fn, "w") as f:
        json.dump(dict(script=SCRIPT, model=a.model, torch=torch.__version__, device=dev,
                       input=a.input,
                       train_class=a.train_class, train_signals=len(tr), lambdas=lambdas,
                       iters=a.iters, batch=a.batch, rho=a.rho, seed=a.seed, summary=summ,
                       signals=rows, extra=extra, dataset=os.path.basename(a.dataset),
                       seconds=time.time() - t0), f, indent=1)
    print(f"\nwrote results/{os.path.basename(fn)} ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
