# file:        cspb_frames_v1.py
# description: frame sets from the CSPB.ML data sets: blind channelization, normalization, square-law derotation and 64-sample frames
# author:      Mitchell A. Thornton <mitch@smu.edu>, ORCID 0000-0003-3559-9511
# license:     MIT (see LICENSE)
"""The CSPB frame sets used by the learned-codec baselines.

For each signal of a CSPB.ML data set (Spooner, cyclostationary.blog): blind
channelization (the occupied band from a Welch spectrum, then an FFT shift and
decimation), normalization to unit power, a square-law carrier estimate (the frequency and
phase of the spectral line of x^2, halved) used to derotate the signal, and framing into
frames of M complex samples. Both the raw and the derotated frames are written, with the
class label and the SNR from the truth record when it is given. The learned codec of
exp_ml_ntc_v4.py trains on the derotated frames by default (--input der), the same
preprocessing the comparison used. The functions are copied unchanged from
the experiment code that produced the published results.

Run (NumPy and SciPy):
  python3 python/cspb_frames_v1.py --root <CSPB.ML.2018R2 dir> --record <truth record> \
      --name cspb2018r2 --groups 1000
  python3 python/cspb_frames_v1.py --root <CSPB.ML.2022R2 dir> --record none --name cspb2022r2
Writes results/ml_dataset_v2_<name>_<tag>.npz (the file name the learned-codec scripts expect).
"""
import argparse, math, os, sys, time
import numpy as np
from scipy.signal import welch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SCRIPT = "cspb_frames_v1"
MIN_FRAMES_AFTER_DECIMATION = 32
MIN_BAND = 4.0 / 1024          # four Welch bins
MIN_FRAMES_TO_CODE = 16


def read_record(path):
    rec = {}
    with open(path) as f:
        for line in f:
            p = line.split()
            if len(p) < 8:
                continue
            try:
                i = int(p[0])
                T0, cfo, beta, up, down, snr = (float(x) for x in p[2:8])
            except ValueError:
                continue
            rec[i] = dict(index=i, mod=p[1], T=T0 * up / down, cfo=cfo, beta=beta,
                          snr=snr, bw=(1 + beta) / (T0 * up / down))
    return rec


def index_files(root):
    idx = {}
    for d, _, files in os.walk(root):
        for fn in files:
            if fn.startswith("signal_") and fn.endswith(".tim"):
                try:
                    idx[int(fn[7:-4])] = os.path.join(d, fn)
                except ValueError:
                    pass
    return idx


def load_tim(path):
    hdr = np.fromfile(path, dtype="<i4", count=2)
    x = np.fromfile(path, dtype="<f4", offset=8)
    z = (x[0::2] + 1j * x[1::2]).astype(complex)
    if hdr[0] != 2 or len(z) != hdr[1]:
        raise ValueError(f"unexpected header {hdr.tolist()} in {path}")
    return z


def blind_band(z, thresh=1.3, smooth=16):
    """Occupied band: Welch PSD (1024 bins), moving average over `smooth` bins,
    white-noise floor from the 25th percentile, occupied bins above `thresh`
    times the floor, and the contiguous run of occupied bins around the PSD
    peak. Returns (centre, width) in cycles per sample."""
    f, P = welch(z, nperseg=1024, return_onesided=False, detrend=False)
    o = np.argsort(f); f, P = f[o], P[o]
    Ps = np.convolve(np.concatenate([P[-smooth:], P, P[:smooth]]),
                     np.ones(smooth) / smooth, "same")[smooth:-smooth]
    floor = np.percentile(Ps, 25)
    occ = Ps > thresh * floor
    k = int(np.argmax(Ps))
    if not occ[k] or occ.all():
        return 0.0, 1.0
    lo = k
    while lo - 1 >= 0 and occ[lo - 1]:
        lo -= 1
    hi = k
    while hi + 1 < len(f) and occ[hi + 1]:
        hi += 1
    df = f[1] - f[0]
    return float((f[lo] + f[hi]) / 2), float(min(1.0, f[hi] - f[lo] + df))


def channelize(z, fc, bw, M=64):
    K = max(1, int(math.floor(1.0 / (1.25 * bw))))
    K = min(K, max(1, len(z) // (M * MIN_FRAMES_AFTER_DECIMATION)))
    if K == 1:
        return z, 1
    n = len(z)
    y = z * np.exp(-2j * np.pi * fc * np.arange(n))
    Z = np.fft.fft(y)
    keep = n // K
    Zs = np.concatenate([Z[:keep // 2], Z[n - keep // 2:]])
    return np.fft.ifft(Zs) * (keep / n) ** 0.5 * math.sqrt(K), K


def frames(z, M):
    n = len(z) // M
    return z[:n * M].reshape(n, M)


def square_law_carrier(x, pad=16):
    """Carrier frequency (cycles/sample) and phase from the line of x^2."""
    n = len(x)
    L = pad * n
    S = np.fft.fft(x * x, L)
    mag = np.abs(S)
    k = int(np.argmax(mag))
    # parabolic refinement on the magnitude
    a, b, c = mag[(k - 1) % L], mag[k], mag[(k + 1) % L]
    den = a - 2 * b + c
    delta = 0.5 * (a - c) / den if abs(den) > 1e-30 else 0.0
    f2 = ((k + delta) / L + 0.5) % 1.0 - 0.5          # frequency of the x^2 line
    t = np.arange(n)
    phase2 = np.angle(np.sum(x * x * np.exp(-2j * np.pi * f2 * t)))
    ratio = float(b / (np.median(mag) + 1e-30))
    return f2 / 2.0, phase2 / 2.0, ratio


def pipeline(z, M):
    fc, bw = blind_band(z)
    if bw < MIN_BAND:
        fc, bw = 0.0, 1.0
    y, K = channelize(z, fc, bw, M)
    y = y / math.sqrt(max(float(np.mean(np.abs(y) ** 2)), 1e-15))
    f_hat, ph, _ = square_law_carrier(y)
    der = y * np.exp(-1j * (2 * np.pi * f_hat * np.arange(len(y)) + ph))
    return frames(y, M), frames(der, M), K, len(y)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--record", required=True, help="truth record path, or none for a blind set")
    ap.add_argument("--name", required=True, help="data set name in the output file name")
    ap.add_argument("--groups", type=int, default=1000)
    ap.add_argument("--max-signals", type=int, default=0, help="blind set only; 0 = all")
    ap.add_argument("--heldout-frac", type=float, default=0.3)
    ap.add_argument("--M", type=int, default=64)
    ap.add_argument("--bits", type=float, default=8.0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    t0 = time.time()
    blind = a.record.lower() == "none"
    rec = {} if blind else read_record(a.record)
    files = index_files(a.root)
    if blind:
        sigs = sorted(files)[: (a.max_signals or None)]
        units = [(i, i, True) for i in sigs]               # (signal, group, held out)
        print(f"{SCRIPT}: blind set, {len(files)} signal files found, exporting {len(sigs)}")
    else:
        avail = [q for q in range(14000) if all(8 * q + k + 1 in files for k in range(8))]
        rng = np.random.default_rng(a.seed)
        grp = np.sort(rng.choice(np.asarray(avail), size=min(a.groups, len(avail)), replace=False))
        held = set(rng.choice(grp, size=int(round(a.heldout_frac * len(grp))), replace=False).tolist())
        units = [(8 * int(q) + k + 1, int(q), int(q) in held) for q in grp for k in range(8)]
        print(f"{SCRIPT}: {len(avail)} complete groups; using {len(grp)}, {len(held)} held out")
    raw, der, meta = [], [], []
    start = 0; skipped = 0
    for n, (i, q, h) in enumerate(units):
        fr, fd, K, L = pipeline(load_tim(files[i]), a.M)
        if len(fr) < MIN_FRAMES_TO_CODE:
            skipped += 1
            continue
        raw.append(fr.astype(np.complex64)); der.append(fd.astype(np.complex64))
        mod, snr = (("unk", float("nan")) if blind else (rec[i]["mod"], rec[i]["snr"]))
        meta.append((i, mod, snr, K, q, h, start, len(fr), len(fr) // 2, 2 * a.bits / L))
        start += len(fr)
        if (n + 1) % 2000 == 0:
            print(f"  {n + 1} / {len(units)} signals ({time.time() - t0:.0f} s)")
    cols = list(zip(*meta))
    tag = (f"blind{len(meta)}" if blind else f"groups{len(units) // 8}_seed{a.seed}")
    fn = os.path.join(ROOT, "results", f"ml_dataset_v2_{a.name}_{tag}.npz")
    os.makedirs(os.path.dirname(fn), exist_ok=True)
    np.savez_compressed(fn, raw=np.concatenate(raw), der=np.concatenate(der),
                        sig_index=np.array(cols[0]), sig_mod=np.array(cols[1]),
                        sig_snr=np.array(cols[2]), sig_K=np.array(cols[3]),
                        sig_group=np.array(cols[4]), sig_heldout=np.array(cols[5]),
                        sig_start=np.array(cols[6]), sig_nfr=np.array(cols[7]),
                        sig_half=np.array(cols[8]), sig_carrier_bps=np.array(cols[9]),
                        M=a.M, bits=a.bits, script=SCRIPT, name=a.name,
                        record=os.path.basename(a.record), root=os.path.abspath(a.root))
    print(f"skipped {skipped}; {start} frames from {len(meta)} signals")
    print(f"wrote results/{os.path.basename(fn)} ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
