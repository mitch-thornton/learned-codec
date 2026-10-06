# file:        plot_v10_figures_v2.py
# description: figures of the widely linear gain against SNR and of rate against the number of training signals
# author:      Mitchell A. Thornton <mitch@smu.edu>, ORCID 0000-0003-3559-9511
# license:     MIT (see LICENSE)
"""Render the v10 figures from results/v10/v10_summary.json (v2: class-trained points are
plotted only where most of the class signals reach the target distortion).

fig_v10_wl_snr: widely linear gain against in-band SNR on CSPB.ML.2018 (BPSK,
MSK, and QPSK as the proper reference), median and 10th to 90th percentile
band per SNR bin.
fig_v10_sample_efficiency: rate at one percent distortion against the number
of same-class training signals for the learned coders (widely linear PCA, and
the class-trained learned codec where its runs exist), with the training-free
AD coder as a horizontal line. Legends sit below the axes, outside the plot box.
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
FIG = os.path.join(ROOT, "figures")
plt.rcParams.update({"font.size": 8, "axes.linewidth": 0.6, "lines.linewidth": 1.1,
                     "font.family": "serif"})


def save(fig, name):
    os.makedirs(FIG, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, f"{name}.{ext}"), bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"wrote figures/{name}.pdf")


def wl_snr(s):
    sp = s.get("spooner")
    if not sp:
        print("spooner summary missing; fig_v10_wl_snr not rendered"); return
    fig, ax = plt.subplots(figsize=(3.4, 2.2))
    for m, lab, col, mk in (("bpsk", "BPSK", "C0", "o"), ("msk", "MSK", "C1", "s"),
                            ("qpsk", "QPSK (proper)", "C7", "^")):
        B = [b for b in sp["classes"][m]["snr_bins"] if b["med"] is not None and b["n"] >= 20]
        x = [(b["lo"] + b["hi"]) / 2 for b in B]
        ax.plot(x, [b["med"] for b in B], marker=mk, ms=3, color=col, label=lab)
        lo = [b["p10"] for b in B]; hi = [b["p90"] for b in B]
        if all(v is not None for v in lo + hi):
            ax.fill_between(x, lo, hi, color=col, alpha=0.15, lw=0)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("in-band SNR (dB)")
    ax.set_ylabel("widely linear gain (bits/sample)")
    ax.grid(alpha=0.3, lw=0.4)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=3, frameon=False)
    save(fig, "fig_v10_wl_snr")


def sample_eff(s):
    pc = s.get("pca")
    if not pc:
        print("pca summary missing; fig_v10_sample_efficiency not rendered"); return
    nc = s.get("ntc_class") or {}
    fig, axes = plt.subplots(1, 2, figsize=(3.5, 1.9), sharey=False)
    for ax, m, lab in ((axes[0], "bpsk", "BPSK"), (axes[1], "msk", "MSK")):
        c = pc["classes"][m]
        n = sorted(int(k) for k in c["curve"])
        ax.semilogx(n, [c["curve"][str(k)] for k in n], "o-", ms=2.5, color="C2",
                    label="PCA, widely linear (ideal rate)")
        ax.axhline(c["ad"], color="C0", lw=1.0, ls="--", label="AD, training-free (ideal rate)")
        pts = []
        for key, d in nc.items():
            cls, nn = key.split("_")
            own = d["classes"].get(m) if d else None
            if cls == m and own and own.get("ad") is not None and own.get("n", 0) >= 150:
                pts.append((d["train_signals"], d["classes"][m]["ntc"], d["classes"][m]["ad"]))
        if pts:
            pts.sort()
            ax.semilogx([p[0] for p in pts], [p[1] for p in pts], "s-", ms=2.5, color="C3",
                        label="learned codec, class-trained (operational)")
            ax.axhline(pts[0][2], color="C3", lw=0.8, ls=":", label="AD (operational)")
        ax.set_title(lab, fontsize=8)
        ax.set_xlabel("training signals")
        ax.grid(alpha=0.3, lw=0.4)
    axes[0].set_ylabel("bits/sample at 1% distortion")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=2, frameon=False,
               fontsize=6.5)
    fig.tight_layout()
    save(fig, "fig_v10_sample_efficiency")


def main():
    s = json.load(open(os.path.join(ROOT, "results", "v10", "v10_summary.json")))
    wl_snr(s)
    sample_eff(s)


if __name__ == "__main__":
    main()
