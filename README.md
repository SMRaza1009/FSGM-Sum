# FSGM-Sum: Factorized Spatio-Temporal Gating Mixer for Efficient Video Summarization

Official PyTorch implementation of **FSGM-Sum**, a lightweight supervised video summarization model that replaces dense self-attention with a factorized temporal and feature-channel gating mixer.

- A **temporal branch** estimates *when* salient events occur through multi-dilation (1, 2, 4) depthwise-separable temporal mixing, complemented by a per-timestep **FFT spectral bias**.
- A **channel branch** estimates *which* feature dimensions carry evidence through temporal mean/max pooling and a bottleneck gating MLP.
- The two estimates are fused by an **additive outer factorization** into a `T x D` gate (cost linear in `T`), calibrated along both axes, and scored by a **temperature-calibrated sigmoid head**.
- Training adds ranking, shot-smoothness, shotwise-alignment, and dual-softmax terms aligned with knapsack-based summary decoding.

FSGM-Sum has **3.43M parameters (13.09 MB)**.

## Contents
- [Environment](#environment)
- [Data](#data)
- [Evaluation protocol](#evaluation-protocol)
- [Pre-trained weights](#pre-trained-weights)
- [Evaluation](#evaluation)
- [Training](#training)
- [Ablations](#ablations)
- [Summaries of raw videos](#summaries-of-raw-videos)
- [Code structure](#code-structure)
- [Citation](#citation)

## Environment

Experiments in the paper were run with:

| GPU | CUDA | Python | PyTorch | torchvision |
|:---:|:---:|:---:|:---:|:---:|
| NVIDIA RTX A6000 (48 GB) | 11.8 | 3.8 | 2.2.1 | 0.17.1 |

```bash
conda create -n fsgm python=3.8.5
conda activate fsgm
git clone https://github.com/SMRaza1009/FSGM-Sum.git
cd FSGM-Sum
pip install torch==2.2.1 torchvision==0.17.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

All commands below are run from the repository root.

## Data

We use the standard pre-extracted GoogLeNet pool5 features (`D = 1024`, one feature every 15 frames) of SumMe and TVSum in HDF5 format, as distributed with [PGL-SUM](https://github.com/e-apostolidis/PGL-SUM). Download the two files and place them in `data/`:

```
data/
├── eccv16_dataset_summe_google_pool5.h5
├── eccv16_dataset_tvsum_google_pool5.h5
└── ydata-anno.tsv        # TVSum per-annotator scores (included in this repository)
```

- [SumMe](https://link.springer.com/chapter/10.1007/978-3-319-10584-0_33): 25 user videos, 15-18 binary user summaries each.
- [TVSum](https://openaccess.thecvf.com/content_cvpr_2015/papers/Song_TVSum_Summarizing_Web_2015_CVPR_paper.pdf): 50 YouTube videos, importance scores from 20 annotators.

## Evaluation protocol

- **Splits.** The exact five train/test splits used in the paper are in `splits/SumMe_splits.txt` and `splits/TVSum_splits.txt` (one line per split: `<dataset>/split<k>/<train videos>/<test videos>`). Each split holds out 20% of the videos for testing. Split `k` in these files corresponds to checkpoint `split{k+1}.pt`.
- **Checkpoint selection.** From each split's training videos, 12.5% are held out as validation videos (a deterministic, seeded choice; 7:1:2 train/validation/test overall). After every epoch the model is evaluated on the validation videos, and the epoch with the highest Spearman's ρ (Kendall's τ as tie-breaker) is kept. **Test videos are never used for model selection**; they are evaluated once, with the selected checkpoint.
- **Metrics.** Kendall's τ and Spearman's ρ. On TVSum, the predicted frame scores are correlated with each annotator and averaged. On SumMe, the scores are first decoded into a keyshot summary (KTS shots, 0/1 knapsack, 15% budget) that is correlated with the averaged user summary. The keyshot F1 (maximum over user summaries on SumMe) is also reported.
- **Seed.** 123456 for PyTorch, NumPy, and Python `random`, with deterministic cuDNN. Some CUDA kernels (e.g. the backward pass of adaptive average pooling) are not bit-deterministic, so retraining can differ from the released weights in the last digit.

## Pre-trained weights

The trained weights for all ten splits (five for SumMe, five for TVSum) are attached to the [GitHub release](https://github.com/SMRaza1009/FSGM-Sum/releases). Download `fsgm_sum_weights.zip` and extract it in the repository root:

```
weights/
├── SumMe/split1.pt ... split5.pt
└── TVSum/split1.pt ... split5.pt
```

## Evaluation

```bash
python inference.py
```

This evaluates every split's checkpoint on its test videos and prints, for each dataset, the per-split scores and their mean ± sample standard deviation. Results are also written to `results/inference_results.json`. Use `--datasets SumMe` or `--datasets TVSum` to evaluate one dataset, `--weights_dir` to point to another set of checkpoints, and `--device cpu` to run without a GPU.

## Training

```bash
python train.py
```

This trains SumMe and then TVSum on all five splits (100 epochs each, Adam, lr `1e-3`, weight decay `1e-7`, one video per step). It saves the selected checkpoint of every split to `weights/<dataset>/split<k>.pt` and the per-split validation/test results with their mean ± sample standard deviation to `results/train_results.json`.

All hyperparameters are command-line options whose defaults are the values in the paper (see `config.py`), e.g.:

| Option | Default | Meaning |
|---|---|---|
| `--fsgm_hidden` | 64 | temporal branch hidden width `D_h` |
| `--fsgm_dilations` | 1 2 4 | dilation rates of the temporal branch |
| `--fsgm_fft_bias` | `timestep` | spectral bias: `timestep`, `scalar`, or `none` |
| `--Dropout_ratio` | 0.6 | dropout on the calibrated attention maps (the scoring head uses 0.5) |
| `--anno_mix_alpha` | 0.5 | multi-annotator target mixing (SumMe) |
| `--lambda_rank`, `--rank_num_pairs` | 0.10, 1024 | pairwise ranking loss |
| `--lambda_shot_smooth` | 0.06 | shot-smoothness loss (SumMe) |
| `--alpha_shotwise` | 0.08 | shotwise alignment loss (SumMe) |
| `--lambda_dual`, `--tau_sharp`, `--tau_smooth`, `--dual_skip_alpha` | 0.08, 0.40, 1.50, 0.15 | dual-softmax loss (SumMe) |
| `--aux_warmup_epochs` | 5 | warmup before the shotwise and dual-softmax terms |
| `--val_ratio` | 0.125 | fraction of training videos held out for checkpoint selection |

## Ablations

Every ablation in the paper is one command. Give each run its own `--weights_dir` and `--results_dir` so the main checkpoints are not overwritten:

```bash
run() { name=$1; shift; python train.py --weights_dir weights_ablation/$name --results_dir results/ablation/$name "$@"; }
```

**Architecture (one factor changed from the full model):**

```bash
run temporal_only   --fsgm_branches temporal
run channel_only    --fsgm_branches channel
run fft_none        --fsgm_fft_bias none
run fft_scalar      --fsgm_fft_bias scalar
run dil_1           --fsgm_dilations 1
run dil_12          --fsgm_dilations 1 2
run dil_1248        --fsgm_dilations 1 2 4 8
run pool_mean       --fsgm_pooling mean
run pool_max        --fsgm_pooling max
run fusion_mul      --fsgm_fusion mul
run temp_fixed      --temp_learnable false
```

**Training objective (one term removed):**

```bash
run no_rank         --rank_loss false
run no_shot_smooth  --shot_smooth_loss false
run no_shotwise     --shotwise_loss false
run no_dual         --dual_softmax_loss false
run mse_only        --rank_loss false --shot_smooth_loss false --shotwise_loss false --dual_softmax_loss false
```

**Step-by-step component addition:**

```bash
run step1_base      --fsgm_branches none --Softmax_axis none --Positional_encoding none --CLS_on false --key_value_emb none --Skip_connection none
run step2_gate      --Softmax_axis none --Positional_encoding none --CLS_on false --key_value_emb none --Skip_connection none
run step3_dualaxis  --Positional_encoding none --CLS_on false --key_value_emb none --Skip_connection none
run step4_pe        --CLS_on false --key_value_emb none --Skip_connection none
run step5_cls       --key_value_emb none --Skip_connection none
run step6_kv        --Skip_connection none
run step7_full
```

## Summaries of raw videos

`generate_video.py` extracts GoogLeNet features from raw videos, segments them into shots with KTS, scores them with a trained FSGM-Sum checkpoint, and writes the summary video:

```bash
# a single video
python generate_video.py --input_is_file true --file_path ./videos/example.mp4 --save_path ./summary_videos --weight_path ./weights/SumMe/split1.pt

# every .mp4 in a directory
python generate_video.py --input_is_file false --dir_path ./videos --ext mp4 --save_path ./summary_videos --weight_path ./weights/SumMe/split1.pt
```

`--sample_rate` (default 15) sets the frame sampling interval. Summaries are written with the `mp4v` codec; change `fourcc` in `produce_video` for other containers.

## Code structure

| File | Purpose |
|---|---|
| `models/FSGM.py` | FSGM-Sum model: factorized gate (`_FSGMAttention`) and full model (`FSGMSum`) |
| `models/positional_encoding.py` | positional encodings |
| `config.py` | all options and their defaults |
| `dataset.py` | split loading, validation hold-out, data loaders |
| `train.py` | training objective and training loop |
| `evaluation_metrics.py` | Kendall's τ, Spearman's ρ, keyshot F1, and the evaluation loop |
| `generate_summary.py`, `knapsack_implementation.py` | shot scoring and 0/1 knapsack decoding |
| `inference.py` | evaluation of trained checkpoints |
| `generate_video.py`, `video_helper.py`, `kts/` | summaries of raw videos |

## Citation

If you use this code, please cite:

```bibtex
@article{raza2026fsgmsum,
  title   = {{FSGM-Sum}: Factorized Spatio-Temporal Gating Mixer for Efficient Video Summarization},
  author  = {Raza, Syed Muhammad and Tariq, Omer},
  journal = {Pattern Recognition},
  note    = {under review},
  year    = {2026}
}
```

## Acknowledgements

The SumMe and TVSum feature files are distributed by [PGL-SUM](https://github.com/e-apostolidis/PGL-SUM). Kernel temporal segmentation (`kts/`) is from [LEAR](http://lear.inrialpes.fr/software), via [DSNet](https://github.com/li-plus/DSNet).

## License

MIT


## License

MIT (see `LICENSE`). The KTS code in `kts/` follows its original license.
