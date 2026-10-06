# Learned-Codec Baselines for Radio-Signal Transform Coding

Mitchell A. Thornton, Darwin Deason Institute for Cyber Security and Department of Electrical
and Computer Engineering, Southern Methodist University.

This repository contains the learned-codec baselines the AD transform coder is compared against,
the code that builds their training and test frames, and the scripts that draw the
learned-codec comparison figures. It does not contain the algebraic-diversity coder itself.

## This repository contains

- `python/cspb_frames_v1.py`: the frame sets from Spooner's CSPB.ML data sets: blind
  channelization (occupied band from a Welch spectrum, FFT shift and decimation),
  normalization, a square-law carrier estimate used to derotate each signal, and frames of
  64 complex samples, raw and derotated, with class labels and SNR from the truth record.
- `python/exp_ml_ntc_v4.py`: the learned codecs (mean-scale hyperprior and factorized prior
  on frames of 64 complex samples), trained without labels on all classes or with labels on
  one class (`--train-class`, `--train-signals`), the architecture variants of the ablation
  (`--hidden`, `--latent`, `--act gdn`, `--deep`, `--context`), and blind scoring on a second
  data set (`--extra-test`).
- `python/ntc_architecture_report_v1.py` (with `exp_ml_ntc_v3.py`): parameters and
  multiply-accumulates per complex sample of the codec, by layer, written to `results/v10/`.
- `python/plot_v10_figures_v2.py` with `results/v10/v10_summary.json`: the
  figures of the widely linear gain against SNR and of rate against the number of training
  signals, drawn from the published results (which include the algebraic-diversity coder's
  results as numbers; its code is not part of this repository).

## Reproduce

    pip install -r requirements.txt
    python3 python/cspb_frames_v1.py --root <CSPB.ML.2018R2 dir> --record <truth record> \
        --name cspb2018r2 --groups 1000
    python3 python/cspb_frames_v1.py --root <CSPB.ML.2022R2 dir> --record none --name cspb2022r2
    python3 python/exp_ml_ntc_v4.py --dataset results/ml_dataset_v2_cspb2018r2_groups1000_seed0.npz
    python3 python/ntc_architecture_report_v1.py
    python3 python/plot_v10_figures_v2.py

The CSPB.ML data sets are distributed by C. M. Spooner (cyclostationary.blog). Training is
far faster on a GPU or on Apple silicon (MPS) than on a CPU. The frame-set code is copied
unchanged from the experiment code that produced the published results and writes the
same frames.

## License

MIT; see LICENSE.
