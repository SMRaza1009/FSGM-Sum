# # BEST OF BEST CODE + Dual-Softmax (SumMe-only, training-only)
import os
import json
import h5py
import numpy as np
import shutil
import torch
import torch.nn.functional as F

from tqdm import tqdm

from config import get_config
from dataset import create_dataloader
from evaluation_metrics import get_corr_coeff
from generate_summary import generate_summary
from model import set_model
from utils import report_params, print_args, get_gt


# ============================================================
# Multi-annotator training (SumMe-only) + summary alignment
# ============================================================

# -----------------------------
# Multi-annotator supervision
# -----------------------------
MULTI_ANNO_ON = True
MULTI_ANNO_SUMME_ONLY = True
ANNO_MIX_ON = True
ANNO_MIX_ALPHA = 0.50

# -----------------------------
# Low-risk add-ons (training-only)
# -----------------------------
RANK_LOSS_ON = True
RANK_NUM_PAIRS = 1024
LAMBDA_RANK = 0.10

SHOT_SMOOTH_ON = True
LAMBDA_SHOT_SMOOTH = 0.06

# -----------------------------
# Summary-aligned auxiliary (SumMe-only)
# -----------------------------
SUMMARY_ALIGN_ON = True
SUMMARY_ALIGN_SUMME_ONLY = True
ALPHA_SHOTWISE_MSE = 0.08
SUMMARY_ALIGN_WARMUP_EPOCHS = 5

# ============================================================
# Phase B0: Dual-Softmax + Residual Skip (SumMe-only, training-only)
# ============================================================
# Purpose: stabilize peak selection while preserving contrast.
# We DO NOT replace output; we add a small auxiliary loss.
DUAL_SOFTMAX_ON = True
DUAL_SOFTMAX_SUMME_ONLY = True
DUAL_SOFTMAX_WARMUP_EPOCHS = 5

TAU_SHARP = 0.40   # 0.25~0.60 (peaky)
TAU_SMOOTH = 1.50  # 1.00~2.50 (stable)
DUAL_SKIP_ALPHA = 0.15 # 0.10 # residual mix strength
LAMBDA_DUAL = 0.08  # 0.05    # small auxiliary weight

# -----------------------------
# Cache SumMe metadata per video
# -----------------------------
_summe_meta_cache = {}

# -----------------------------
# Debug
# -----------------------------
DEBUG = True
DEBUG_FAIL_FAST = True
DEBUG_EVERY_N_EPOCHS = 1
DEBUG_SAVE_SAMPLES = True
DEBUG_SAVE_MAX = 5


def safe_nanmean(x):
    if x is None or len(x) == 0:
        return np.nan
    return float(np.nanmean(np.asarray(x, dtype=np.float64)))


def tensor_stats(t: torch.Tensor, name: str):
    if t is None:
        return {"name": name, "status": "None"}
    x = t.detach()
    finite = torch.isfinite(x)
    out = {
        "name": name,
        "shape": list(x.shape),
        "dtype": str(x.dtype),
        "device": str(x.device),
        "numel": int(x.numel()),
        "finite": int(finite.sum().item()),
        "nan": int(torch.isnan(x).sum().item()),
        "inf": int(torch.isinf(x).sum().item()),
    }
    if finite.any():
        xf = x[finite]
        out.update({
            "min": float(xf.min().item()),
            "max": float(xf.max().item()),
            "mean": float(xf.mean().item()),
            "std": float(xf.std(unbiased=False).item()) if xf.numel() > 1 else 0.0,
        })
    return out


def debug_print(title: str, payload: dict):
    print(f"\n[DEBUG] {title}")
    for k, v in payload.items():
        print(f"  - {k}: {v}")


def save_json(path: str, obj: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def normalize_video_num(video_num):
    """Make video_num a clean string key for H5 access."""
    if isinstance(video_num, (list, tuple)) and len(video_num) == 1:
        video_num = video_num[0]
    if isinstance(video_num, np.ndarray) and video_num.size == 1:
        video_num = video_num.item()
    if isinstance(video_num, bytes):
        video_num = video_num.decode("utf-8")
    return str(video_num)


# ============================
# Losses
# ============================

def pairwise_rank_loss(y_pred: torch.Tensor, y_true: torch.Tensor, num_pairs: int = 512) -> torch.Tensor:
    """Differentiable ranking loss aligned with Spearman/Kendall."""
    y_pred = y_pred.view(-1)
    y_true = y_true.view(-1)
    T = y_pred.numel()
    if T < 2:
        return y_pred.new_tensor(0.0)

    i = torch.randint(0, T, (num_pairs,), device=y_pred.device)
    j = torch.randint(0, T, (num_pairs,), device=y_pred.device)

    neq = (i != j)
    i, j = i[neq], j[neq]
    if i.numel() == 0:
        return y_pred.new_tensor(0.0)

    diff_true = y_true[i] - y_true[j]
    sign = torch.sign(diff_true)

    mask = sign != 0
    if not mask.any():
        return y_pred.new_tensor(0.0)

    sign = sign[mask]
    diff_pred = (y_pred[i] - y_pred[j])[mask]

    return F.softplus(-diff_pred * sign).mean()


def summe_shot_smooth_loss(y_pred: torch.Tensor, change_points: np.ndarray, picks: np.ndarray) -> torch.Tensor:
    """Encourage within-shot smoothness by minimizing variance within each shot."""
    y_pred = y_pred.view(-1)
    if y_pred.numel() < 2:
        return y_pred.new_tensor(0.0)

    device = y_pred.device
    picks_t = torch.from_numpy(picks).to(device=device)

    losses = []
    for (s, e) in change_points:
        m = (picks_t >= int(s)) & (picks_t <= int(e))
        if int(m.sum().item()) >= 2:
            seg = y_pred[m]
            losses.append(seg.var(unbiased=False))

    if len(losses) == 0:
        return y_pred.new_tensor(0.0)
    return torch.stack(losses).mean()


# ============================
# Multi-annotator target helpers
# ============================

def _align_1d_tensor_to(t: torch.Tensor, target_len: int) -> torch.Tensor:
    """Pad/truncate 1D tensor to target_len (repeat last value if padding)."""
    t = t.view(-1)
    n = int(t.numel())
    if n == target_len:
        return t
    m = min(n, target_len)
    t2 = t[:m]
    if m < target_len:
        pad = t2[-1:].repeat(target_len - m)
        t2 = torch.cat([t2, pad], dim=0)
    return t2


def sample_summe_user_summary_target(video_num: str, device: torch.device) -> torch.Tensor:
    """Sample one user_summary row and map to picks => length matches model output."""
    if video_num not in _summe_meta_cache or "user_summary" not in _summe_meta_cache[video_num]:
        with h5py.File("./data/eccv16_dataset_summe_google_pool5.h5", "r") as hdf:
            cp = np.array(hdf[f"{video_num}/change_points"], dtype=np.int64)
            pk = np.array(hdf[f"{video_num}/picks"], dtype=np.int64)
            us = np.array(hdf[f"{video_num}/user_summary"], dtype=np.float32)
        _summe_meta_cache[video_num] = {"cp": cp, "pk": pk, "user_summary": us}

    us = _summe_meta_cache[video_num]["user_summary"]
    pk = _summe_meta_cache[video_num]["pk"]

    U = int(us.shape[0])
    u = np.random.randint(0, U)
    ann_frames = us[u]
    ann_t = ann_frames[pk]
    return torch.from_numpy(np.asarray(ann_t, dtype=np.float32)).to(device=device)


# ============================
# Summary-aligned auxiliary (SumMe)
# ============================

def _zscore(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    m = x.mean()
    s = x.std(unbiased=False)
    return (x - m) / (s + eps)


def _shot_scores_from_picks(y: torch.Tensor, change_points: np.ndarray, picks: np.ndarray) -> torch.Tensor:
    """Aggregate picked-frame predictions into shot scores by mean pooling."""
    y = y.view(-1)
    device = y.device
    picks_t = torch.from_numpy(picks).to(device=device)

    shot_scores = []
    gmean = y.mean()
    for (s, e) in change_points:
        m = (picks_t >= int(s)) & (picks_t <= int(e))
        if int(m.sum().item()) >= 1:
            shot_scores.append(y[m].mean())
        else:
            shot_scores.append(gmean)
    return torch.stack(shot_scores, dim=0)


def summe_shotwise_mse_aux(output: torch.Tensor, video_num: str, device: torch.device) -> torch.Tensor:
    """Shotwise trend alignment: MSE between normalized predicted shot scores and
    normalized GT shot probability (avg over users), SumMe-only."""
    if video_num not in _summe_meta_cache or "cp" not in _summe_meta_cache[video_num]:
        with h5py.File("./data/eccv16_dataset_summe_google_pool5.h5", "r") as hdf:
            cp = np.array(hdf[f"{video_num}/change_points"], dtype=np.int64)
            pk = np.array(hdf[f"{video_num}/picks"], dtype=np.int64)
            us = np.array(hdf[f"{video_num}/user_summary"], dtype=np.float32)
        _summe_meta_cache[video_num] = {"cp": cp, "pk": pk, "user_summary": us}

    cp = _summe_meta_cache[video_num]["cp"]
    pk = _summe_meta_cache[video_num]["pk"]
    us = _summe_meta_cache[video_num]["user_summary"]

    shot_pred = _shot_scores_from_picks(output, cp, pk)

    gt_list = []
    for (s, e) in cp:
        seg = us[:, int(s):int(e) + 1]
        gt_list.append(float(seg.mean()))
    shot_gt = torch.tensor(gt_list, device=device, dtype=torch.float32)

    shot_pred_n = _zscore(shot_pred)
    shot_gt_n = _zscore(shot_gt)

    return F.mse_loss(shot_pred_n, shot_gt_n)


# ============================
# Dual-Softmax calibration (SumMe)
# ============================

def _safe_logit(p: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    p = p.clamp(min=eps, max=1.0 - eps)
    return torch.log(p) - torch.log1p(-p)


def dual_softmax_residual_mix(scores_01: torch.Tensor,
                              tau_sharp: float = 0.40,
                              tau_smooth: float = 1.50,
                              alpha: float = 0.10,
                              eps: float = 1e-6) -> torch.Tensor:
    """scores_01: [T] in [0,1]. Returns [T] in [0,1].

    Steps:
    - convert to logits
    - compute two softmax distributions over time (sharp + smooth)
    - multiply and renormalize (dual-softmax)
    - map back to [0,1] via sigmoid of normalized log-prob
    - residual skip mix with original scores
    """
    s = scores_01.view(-1)
    if s.numel() < 2:
        return s

    z = _safe_logit(s, eps=eps)  # [T]

    p_sharp = F.softmax(z / max(eps, float(tau_sharp)), dim=0)
    p_smooth = F.softmax(z / max(eps, float(tau_smooth)), dim=0)

    p = p_sharp * p_smooth
    p = p / (p.sum() + eps)

    # log-prob -> normalize -> sigmoid to get back to [0,1]
    z_dual = torch.log(p + eps)
    z_dual = _zscore(z_dual)
    s_dual = torch.sigmoid(z_dual)

    a = float(alpha)
    return ((1.0 - a) * s + a * s_dual).clamp(0.0, 1.0)


# =====================
# Load config
# =====================
config = get_config()
print_args(config)

os.makedirs("./tmp", exist_ok=True)
os.makedirs("./weights", exist_ok=True)
os.makedirs("./debug_preds", exist_ok=True)

report_params(
    model_name=config.model_name,
    Scale=config.Scale,
    Softmax_axis=config.Softmax_axis,
    Balance=config.Balance,
    Positional_encoding=config.Positional_encoding,
    Positional_encoding_shape=config.Positional_encoding_shape,
    Positional_encoding_way=config.Positional_encoding_way,
    Dropout_on=config.Dropout_on,
    Dropout_ratio=config.Dropout_ratio,
    Classifier_on=config.Classifier_on,
    CLS_on=config.CLS_on,
    CLS_mix=config.CLS_mix,
    key_value_emb=config.key_value_emb,
    Skip_connection=config.Skip_connection,
    Layernorm=config.Layernorm
)

if not hasattr(config, "datasets"):
    config.datasets = ["SumMe", "TVSum"]


# =====================
# Training
# =====================
for dataset in tqdm(config.datasets, total=len(config.datasets), ncols=70, leave=True, position=0):
    os.makedirs(f"./weights/{dataset}", exist_ok=True)

    user_scores = get_gt(dataset)

    if dataset == 'SumMe':
        batch_size = 1 if config.batch_size == '1' else int(config.SumMe_len * 0.8 * float(config.batch_size))
    elif dataset == 'TVSum':
        batch_size = 1 if config.batch_size == '1' else int(config.TVSum_len * 0.8 * float(config.batch_size))
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    for split_id, (train_loader, test_loader) in tqdm(
        enumerate(create_dataloader(dataset)),
        total=5, ncols=70, leave=False, position=1, desc=dataset
    ):
        tmp_ckpt = "./tmp/weight.pt"
        if os.path.exists(tmp_ckpt):
            try:
                os.remove(tmp_ckpt)
            except Exception:
                pass

        model = set_model(
            model_name=config.model_name,
            Scale=config.Scale,
            Softmax_axis=config.Softmax_axis,
            Balance=config.Balance,
            Positional_encoding=config.Positional_encoding,
            Positional_encoding_shape=config.Positional_encoding_shape,
            Positional_encoding_way=config.Positional_encoding_way,
            Dropout_on=config.Dropout_on,
            Dropout_ratio=config.Dropout_ratio,
            Classifier_on=config.Classifier_on,
            CLS_on=config.CLS_on,
            CLS_mix=config.CLS_mix,
            key_value_emb=config.key_value_emb,
            Skip_connection=config.Skip_connection,
            Layernorm=config.Layernorm
        ).to(config.device)

        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=float(config.learning_rate),
            weight_decay=float(config.weight_decay)
        )

        best_spear = -1e9
        best_kendall = -1e9
        saved_samples = 0

        for epoch in tqdm(
            range(config.epochs),
            total=config.epochs, ncols=70, leave=False, position=2,
            desc=f"Split{split_id+1}"
        ):
            # ------------------ TRAIN ------------------
            model.train()
            update_loss = None
            batch = 0

            for feature, gtscore, dataset_name, video_num in tqdm(
                train_loader, ncols=70, leave=False, position=3, desc=f"Epoch{epoch+1}_TRAIN"
            ):
                dataset_name = str(dataset_name)
                video_num = normalize_video_num(video_num)

                feature = feature.to(config.device)
                gtscore = gtscore.to(config.device).view(-1)

                output = model(feature).view(-1)  # [T] in [0,1] (FSGM uses sigmoid)

                if DEBUG:
                    if not torch.isfinite(output).all():
                        debug_print("NON-FINITE OUTPUT (TRAIN)", {
                            "dataset": dataset,
                            "split": split_id + 1,
                            "epoch": epoch + 1,
                            "video_num": video_num,
                            "output": tensor_stats(output, "output"),
                            "gtscore": tensor_stats(gtscore, "gtscore"),
                        })
                        if DEBUG_FAIL_FAST:
                            raise FloatingPointError("NaN/Inf detected in model output during TRAIN")

                    if output.shape != gtscore.shape:
                        debug_print("SHAPE MISMATCH (TRAIN)", {
                            "video_num": video_num,
                            "output_shape": list(output.shape),
                            "gtscore_shape": list(gtscore.shape),
                        })
                        if DEBUG_FAIL_FAST:
                            raise ValueError("Output and GT shapes differ during TRAIN")

                # -----------------------------
                # Multi-annotator target (SumMe ONLY)
                # -----------------------------
                target = gtscore
                ann = None

                if MULTI_ANNO_ON and (not MULTI_ANNO_SUMME_ONLY or dataset_name == "SumMe"):
                    if dataset_name == "SumMe":
                        ann = sample_summe_user_summary_target(video_num, config.device)
                        ann = _align_1d_tensor_to(ann, int(output.numel()))

                        if ANNO_MIX_ON:
                            target = (1.0 - ANNO_MIX_ALPHA) * gtscore + ANNO_MIX_ALPHA * ann
                        else:
                            target = ann

                # Base regression loss
                loss = criterion(output, target)

                # Rank regularizer
                if RANK_LOSS_ON:
                    loss_rank = pairwise_rank_loss(output, target, num_pairs=RANK_NUM_PAIRS)
                    loss = loss + (LAMBDA_RANK * loss_rank)

                # Shot smoothness (SumMe only)
                if SHOT_SMOOTH_ON and dataset_name == "SumMe":
                    if video_num not in _summe_meta_cache or "cp" not in _summe_meta_cache[video_num]:
                        with h5py.File("./data/eccv16_dataset_summe_google_pool5.h5", "r") as hdf:
                            cp = np.array(hdf[f"{video_num}/change_points"], dtype=np.int64)
                            pk = np.array(hdf[f"{video_num}/picks"], dtype=np.int64)
                        if video_num not in _summe_meta_cache:
                            _summe_meta_cache[video_num] = {"cp": cp, "pk": pk}
                        else:
                            _summe_meta_cache[video_num]["cp"] = cp
                            _summe_meta_cache[video_num]["pk"] = pk

                    cp = _summe_meta_cache[video_num]["cp"]
                    pk = _summe_meta_cache[video_num]["pk"]
                    loss_smooth = summe_shot_smooth_loss(output, cp, pk)
                    loss = loss + (LAMBDA_SHOT_SMOOTH * loss_smooth)

                # Summary-aligned auxiliary (SumMe only, after warmup)
                if SUMMARY_ALIGN_ON and (not SUMMARY_ALIGN_SUMME_ONLY or dataset_name == "SumMe"):
                    if dataset_name == "SumMe" and epoch >= SUMMARY_ALIGN_WARMUP_EPOCHS:
                        loss_shot = summe_shotwise_mse_aux(output, video_num, device=config.device)
                        loss = loss + (ALPHA_SHOTWISE_MSE * loss_shot)

                # -------------------------------------------------
                # Dual-Softmax + Residual Skip auxiliary (SumMe only)
                # -------------------------------------------------
                if DUAL_SOFTMAX_ON and (not DUAL_SOFTMAX_SUMME_ONLY or dataset_name == "SumMe"):
                    if dataset_name == "SumMe" and epoch >= DUAL_SOFTMAX_WARMUP_EPOCHS:
                        out_dual = dual_softmax_residual_mix(
                            output,
                            tau_sharp=TAU_SHARP,
                            tau_smooth=TAU_SMOOTH,
                            alpha=DUAL_SKIP_ALPHA,
                        )
                        loss_dual = criterion(out_dual, target)
                        loss = loss + (LAMBDA_DUAL * loss_dual)

                if DEBUG and epoch == 0 and batch == 0:
                    dbg = {
                        "base_mse": float(criterion(output, target).detach().cpu()),
                        "target_stats": tensor_stats(target, "target"),
                        "output_stats": tensor_stats(output, "output"),
                    }
                    if ann is not None:
                        dbg["anno_mix_alpha"] = float(ANNO_MIX_ALPHA) if ANNO_MIX_ON else 1.0
                        dbg["ann_stats"] = {
                            "min": float(ann.min().item()),
                            "max": float(ann.max().item()),
                            "mean": float(ann.mean().item()),
                        }
                    if RANK_LOSS_ON:
                        dbg["rank"] = float(loss_rank.detach().cpu())
                    if SHOT_SMOOTH_ON and dataset_name == "SumMe":
                        dbg["shot"] = float(loss_smooth.detach().cpu())
                    if SUMMARY_ALIGN_ON and dataset_name == "SumMe" and epoch >= SUMMARY_ALIGN_WARMUP_EPOCHS:
                        dbg["shotwise_aux"] = float(loss_shot.detach().cpu())
                    if DUAL_SOFTMAX_ON and dataset_name == "SumMe" and epoch >= DUAL_SOFTMAX_WARMUP_EPOCHS:
                        dbg["dual_aux"] = float(loss_dual.detach().cpu())
                        dbg["dual_params"] = {
                            "tau_sharp": TAU_SHARP,
                            "tau_smooth": TAU_SMOOTH,
                            "alpha": DUAL_SKIP_ALPHA,
                            "lambda": LAMBDA_DUAL,
                        }
                        dbg["out_dual_stats"] = tensor_stats(out_dual, "out_dual")
                    debug_print("LOSS COMPONENTS (first batch)", dbg)

                # Accumulate micro-batch
                update_loss = loss if update_loss is None else (update_loss + loss)
                batch += 1

                if batch == batch_size:
                    optimizer.zero_grad()
                    (update_loss / batch).backward()
                    optimizer.step()
                    update_loss = None
                    batch = 0

            # leftover
            if batch > 0 and update_loss is not None:
                optimizer.zero_grad()
                (update_loss / batch).backward()
                optimizer.step()

            # ------------------ VALIDATE ------------------
            val_spears, val_kendalls = [], []
            model.eval()

            with torch.no_grad():
                for feature, gtscore, dataset_name, video_num in tqdm(
                    test_loader, ncols=70, leave=False, position=3, desc=f"Epoch{epoch+1}_TEST"
                ):
                    dataset_name = str(dataset_name)
                    video_num = normalize_video_num(video_num)

                    feature = feature.to(config.device)
                    output = model(feature).view(-1)

                    if DEBUG and (not torch.isfinite(output).all()):
                        debug_print("NON-FINITE OUTPUT (VAL)", {
                            "dataset": dataset,
                            "split": split_id + 1,
                            "epoch": epoch + 1,
                            "video_num": video_num,
                            "output": tensor_stats(output, "output"),
                        })
                        if DEBUG_FAIL_FAST:
                            raise FloatingPointError("NaN/Inf detected in model output during VAL")

                    if dataset_name in ['SumMe', 'TVSum']:
                        h5_path = f'./data/eccv16_dataset_{dataset_name.lower()}_google_pool5.h5'
                        with h5py.File(h5_path, 'r') as hdf:
                            user_summary = np.array(hdf[video_num]['user_summary'])
                            sb = np.array(hdf[f"{video_num}/change_points"])
                            n_frames = int(np.array(hdf[f"{video_num}/n_frames"]))
                            positions = np.array(hdf[f"{video_num}/picks"])

                        scores = output.detach().cpu().numpy().tolist()

                        # IMPORTANT: SumMe validation uses generate_summary alignment
                        summary = generate_summary([sb], [scores], [n_frames], [positions])[0]
                        summary = np.asarray(summary)

                        if dataset_name == 'SumMe':
                            spear, kendall = get_corr_coeff([summary], [video_num], dataset_name, user_summary)
                        else:
                            spear, kendall = get_corr_coeff([scores], [video_num], dataset_name, user_scores)

                        val_spears.append(spear)
                        val_kendalls.append(kendall)

                        if DEBUG and DEBUG_SAVE_SAMPLES and saved_samples < DEBUG_SAVE_MAX:
                            dbg_obj = {
                                "dataset": dataset,
                                "split": int(split_id + 1),
                                "epoch": int(epoch + 1),
                                "video_num": video_num,
                                "scores_min": float(np.min(scores)) if len(scores) else None,
                                "scores_max": float(np.max(scores)) if len(scores) else None,
                                "scores_mean": float(np.mean(scores)) if len(scores) else None,
                                "scores_head": scores[:20],
                                "scores_tail": scores[-20:],
                                "picks_len": int(len(positions)),
                                "n_frames": int(n_frames),
                            }
                            out_path = f"./debug_preds/{dataset}/split{split_id+1}/epoch{epoch+1}_{video_num.replace('/', '_')}.json"
                            save_json(out_path, dbg_obj)
                            saved_samples += 1

            mk = safe_nanmean(val_kendalls)
            ms = safe_nanmean(val_spears)

            if DEBUG and ((epoch + 1) % DEBUG_EVERY_N_EPOCHS == 0):
                debug_print("EPOCH VAL SUMMARY", {
                    "dataset": dataset,
                    "split": split_id + 1,
                    "epoch": epoch + 1,
                    "val_count": int(len(val_spears)),
                    "mean_spear": ms,
                    "mean_kendall": mk,
                    "best_spear": float(best_spear),
                    "best_kendall": float(best_kendall),
                })

            # Save best: prioritize Spearman, tie-break Kendall
            if np.isfinite(ms) and np.isfinite(mk):
                improved = (ms > best_spear) or (ms == best_spear and mk > best_kendall)
                if improved:
                    best_spear = float(ms)
                    best_kendall = float(mk)
                    torch.save(model.state_dict(), tmp_ckpt)

        # Guarantee checkpoint exists
        if not os.path.exists(tmp_ckpt):
            if DEBUG:
                debug_print("NO BEST CKPT SAVED; FALLING BACK TO FINAL", {
                    "dataset": dataset,
                    "split": split_id + 1,
                    "best_spear": float(best_spear),
                    "best_kendall": float(best_kendall),
                })
            torch.save(model.state_dict(), tmp_ckpt)

        dst_ckpt = f'./weights/{dataset}/split{split_id+1}.pt'
        shutil.move(tmp_ckpt, dst_ckpt)









# import os
# import json
# import h5py
# import numpy as np
# import shutil
# import torch
# import torch.nn.functional as F

# from tqdm import tqdm

# from config import get_config
# from dataset import create_dataloader
# from evaluation_metrics import get_corr_coeff
# from generate_summary import generate_summary
# from model import set_model
# from utils import report_params, print_args, get_gt


# # ============================================================
# # Multi-annotator training (SumMe-only) + summary alignment
# # ============================================================

# # -----------------------------
# # Multi-annotator supervision
# # -----------------------------
# MULTI_ANNO_ON = True
# MULTI_ANNO_SUMME_ONLY = True
# ANNO_MIX_ON = True
# ANNO_MIX_ALPHA = 0.30 # 0.50

# # -----------------------------
# # Low-risk add-ons (training-only)
# # -----------------------------
# RANK_LOSS_ON = True
# RANK_NUM_PAIRS = 512 # 1024
# LAMBDA_RANK = 0.05 # 0.10

# SHOT_SMOOTH_ON = False # True
# LAMBDA_SHOT_SMOOTH = 0.06

# # -----------------------------
# # Summary-aligned auxiliary (SumMe-only)
# # -----------------------------
# SUMMARY_ALIGN_ON = False # True
# SUMMARY_ALIGN_SUMME_ONLY = True
# ALPHA_SHOTWISE_MSE = 0.08
# SUMMARY_ALIGN_WARMUP_EPOCHS = 5

# # ============================================================
# # Phase B0: Dual-Softmax + Residual Skip (SumMe-only, training-only)
# # ============================================================
# # Purpose: stabilize peak selection while preserving contrast.
# # We DO NOT replace output; we add a small auxiliary loss.
# DUAL_SOFTMAX_ON = False # True
# DUAL_SOFTMAX_SUMME_ONLY = True
# DUAL_SOFTMAX_WARMUP_EPOCHS = 5

# TAU_SHARP = 0.40   # 0.25~0.60 (peaky)
# TAU_SMOOTH = 1.50  # 1.00~2.50 (stable)
# DUAL_SKIP_ALPHA = 0.15 # residual mix strength
# LAMBDA_DUAL = 0.08  # small auxiliary weight

# # -----------------------------
# # Cache SumMe metadata per video
# # -----------------------------
# _summe_meta_cache = {}

# # -----------------------------
# # Debug
# # -----------------------------
# DEBUG = True
# DEBUG_FAIL_FAST = True
# DEBUG_EVERY_N_EPOCHS = 1
# DEBUG_SAVE_SAMPLES = True
# DEBUG_SAVE_MAX = 5


# def safe_nanmean(x):
#     if x is None or len(x) == 0:
#         return np.nan
#     return float(np.nanmean(np.asarray(x, dtype=np.float64)))


# def tensor_stats(t: torch.Tensor, name: str):
#     if t is None:
#         return {"name": name, "status": "None"}
#     x = t.detach()
#     finite = torch.isfinite(x)
#     out = {
#         "name": name,
#         "shape": list(x.shape),
#         "dtype": str(x.dtype),
#         "device": str(x.device),
#         "numel": int(x.numel()),
#         "finite": int(finite.sum().item()),
#         "nan": int(torch.isnan(x).sum().item()),
#         "inf": int(torch.isinf(x).sum().item()),
#     }
#     if finite.any():
#         xf = x[finite]
#         out.update({
#             "min": float(xf.min().item()),
#             "max": float(xf.max().item()),
#             "mean": float(xf.mean().item()),
#             "std": float(xf.std(unbiased=False).item()) if xf.numel() > 1 else 0.0,
#         })
#     return out


# def debug_print(title: str, payload: dict):
#     print(f"\n[DEBUG] {title}")
#     for k, v in payload.items():
#         print(f"  - {k}: {v}")


# def save_json(path: str, obj: dict):
#     d = os.path.dirname(path)
#     if d:
#         os.makedirs(d, exist_ok=True)
#     with open(path, "w") as f:
#         json.dump(obj, f, indent=2)


# def normalize_video_num(video_num):
#     """Make video_num a clean string key for H5 access."""
#     if isinstance(video_num, (list, tuple)) and len(video_num) == 1:
#         video_num = video_num[0]
#     if isinstance(video_num, np.ndarray) and video_num.size == 1:
#         video_num = video_num.item()
#     if isinstance(video_num, bytes):
#         video_num = video_num.decode("utf-8")
#     return str(video_num)


# def normalize_dataset_name(dataset_name):
#     """Safely normalize dataset_name from dataloader (may arrive as list/ndarray/bytes)."""
#     if isinstance(dataset_name, (list, tuple)) and len(dataset_name) == 1:
#         dataset_name = dataset_name[0]
#     if isinstance(dataset_name, np.ndarray) and dataset_name.size == 1:
#         dataset_name = dataset_name.item()
#     if isinstance(dataset_name, bytes):
#         dataset_name = dataset_name.decode("utf-8")
#     dataset_name = str(dataset_name)
#     # common accidental wrappers
#     dataset_name = dataset_name.replace("('", "").replace("',)", "").replace('"', '').strip()
#     return dataset_name


# # ============================
# # Losses
# # ============================

# def pairwise_rank_loss(y_pred: torch.Tensor, y_true: torch.Tensor, num_pairs: int = 512) -> torch.Tensor:
#     """Differentiable ranking loss aligned with Spearman/Kendall."""
#     y_pred = y_pred.view(-1)
#     y_true = y_true.view(-1)
#     T = y_pred.numel()
#     if T < 2:
#         return y_pred.new_tensor(0.0)

#     i = torch.randint(0, T, (num_pairs,), device=y_pred.device)
#     j = torch.randint(0, T, (num_pairs,), device=y_pred.device)

#     neq = (i != j)
#     i, j = i[neq], j[neq]
#     if i.numel() == 0:
#         return y_pred.new_tensor(0.0)

#     diff_true = y_true[i] - y_true[j]
#     sign = torch.sign(diff_true)

#     mask = sign != 0
#     if not mask.any():
#         return y_pred.new_tensor(0.0)

#     sign = sign[mask]
#     diff_pred = (y_pred[i] - y_pred[j])[mask]

#     return F.softplus(-diff_pred * sign).mean()


# def summe_shot_smooth_loss(y_pred: torch.Tensor, change_points: np.ndarray, picks: np.ndarray) -> torch.Tensor:
#     """Encourage within-shot smoothness by minimizing variance within each shot."""
#     y_pred = y_pred.view(-1)
#     if y_pred.numel() < 2:
#         return y_pred.new_tensor(0.0)

#     device = y_pred.device
#     picks_t = torch.from_numpy(picks).to(device=device)

#     losses = []
#     for (s, e) in change_points:
#         m = (picks_t >= int(s)) & (picks_t <= int(e))
#         if int(m.sum().item()) >= 2:
#             seg = y_pred[m]
#             losses.append(seg.var(unbiased=False))

#     if len(losses) == 0:
#         return y_pred.new_tensor(0.0)
#     return torch.stack(losses).mean()


# # ============================
# # Multi-annotator target helpers
# # ============================

# def _align_1d_tensor_to(t: torch.Tensor, target_len: int) -> torch.Tensor:
#     """Pad/truncate 1D tensor to target_len (repeat last value if padding)."""
#     t = t.view(-1)
#     n = int(t.numel())
#     if n == target_len:
#         return t
#     m = min(n, target_len)
#     t2 = t[:m]
#     if m < target_len:
#         pad = t2[-1:].repeat(target_len - m)
#         t2 = torch.cat([t2, pad], dim=0)
#     return t2


# def sample_summe_user_summary_target(video_num: str, device: torch.device) -> torch.Tensor:
#     """Sample one user_summary row and map to picks => length matches model output."""
#     if video_num not in _summe_meta_cache or "user_summary" not in _summe_meta_cache[video_num]:
#         with h5py.File("./data/eccv16_dataset_summe_google_pool5.h5", "r") as hdf:
#             cp = np.array(hdf[f"{video_num}/change_points"], dtype=np.int64)
#             pk = np.array(hdf[f"{video_num}/picks"], dtype=np.int64)
#             us = np.array(hdf[f"{video_num}/user_summary"], dtype=np.float32)
#         _summe_meta_cache[video_num] = {"cp": cp, "pk": pk, "user_summary": us}

#     us = _summe_meta_cache[video_num]["user_summary"]
#     pk = _summe_meta_cache[video_num]["pk"]

#     U = int(us.shape[0])
#     u = np.random.randint(0, U)
#     ann_frames = us[u]
#     ann_t = ann_frames[pk]
#     return torch.from_numpy(np.asarray(ann_t, dtype=np.float32)).to(device=device)


# # ============================
# # Summary-aligned auxiliary (SumMe)
# # ============================

# def _zscore(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
#     m = x.mean()
#     s = x.std(unbiased=False)
#     return (x - m) / (s + eps)


# def _shot_scores_from_picks(y: torch.Tensor, change_points: np.ndarray, picks: np.ndarray) -> torch.Tensor:
#     """Aggregate picked-frame predictions into shot scores by mean pooling."""
#     y = y.view(-1)
#     device = y.device
#     picks_t = torch.from_numpy(picks).to(device=device)

#     shot_scores = []
#     gmean = y.mean()
#     for (s, e) in change_points:
#         m = (picks_t >= int(s)) & (picks_t <= int(e))
#         if int(m.sum().item()) >= 1:
#             shot_scores.append(y[m].mean())
#         else:
#             shot_scores.append(gmean)
#     return torch.stack(shot_scores, dim=0)


# def summe_shotwise_mse_aux(output: torch.Tensor, video_num: str, device: torch.device) -> torch.Tensor:
#     """Shotwise trend alignment: MSE between normalized predicted shot scores and
#     normalized GT shot probability (avg over users), SumMe-only."""
#     if video_num not in _summe_meta_cache or "cp" not in _summe_meta_cache[video_num]:
#         with h5py.File("./data/eccv16_dataset_summe_google_pool5.h5", "r") as hdf:
#             cp = np.array(hdf[f"{video_num}/change_points"], dtype=np.int64)
#             pk = np.array(hdf[f"{video_num}/picks"], dtype=np.int64)
#             us = np.array(hdf[f"{video_num}/user_summary"], dtype=np.float32)
#         _summe_meta_cache[video_num] = {"cp": cp, "pk": pk, "user_summary": us}

#     cp = _summe_meta_cache[video_num]["cp"]
#     pk = _summe_meta_cache[video_num]["pk"]
#     us = _summe_meta_cache[video_num]["user_summary"]

#     shot_pred = _shot_scores_from_picks(output, cp, pk)

#     gt_list = []
#     for (s, e) in cp:
#         seg = us[:, int(s):int(e) + 1]
#         gt_list.append(float(seg.mean()))
#     shot_gt = torch.tensor(gt_list, device=device, dtype=torch.float32)

#     shot_pred_n = _zscore(shot_pred)
#     shot_gt_n = _zscore(shot_gt)

#     return F.mse_loss(shot_pred_n, shot_gt_n)


# # ============================
# # Dual-Softmax calibration (SumMe)
# # ============================

# def _safe_logit(p: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
#     p = p.clamp(min=eps, max=1.0 - eps)
#     return torch.log(p) - torch.log1p(-p)


# def dual_softmax_residual_mix(scores_01: torch.Tensor,
#                               tau_sharp: float = 0.40,
#                               tau_smooth: float = 1.50,
#                               alpha: float = 0.10,
#                               eps: float = 1e-6) -> torch.Tensor:
#     """scores_01: [T] in [0,1]. Returns [T] in [0,1].

#     Steps:
#     - convert to logits
#     - compute two softmax distributions over time (sharp + smooth)
#     - multiply and renormalize (dual-softmax)
#     - map back to [0,1] via sigmoid of normalized log-prob
#     - residual skip mix with original scores
#     """
#     s = scores_01.view(-1)
#     if s.numel() < 2:
#         return s

#     z = _safe_logit(s, eps=eps)  # [T]

#     p_sharp = F.softmax(z / max(eps, float(tau_sharp)), dim=0)
#     p_smooth = F.softmax(z / max(eps, float(tau_smooth)), dim=0)

#     p = p_sharp * p_smooth
#     p = p / (p.sum() + eps)

#     # log-prob -> normalize -> sigmoid to get back to [0,1]
#     z_dual = torch.log(p + eps)
#     z_dual = _zscore(z_dual)
#     s_dual = torch.sigmoid(z_dual)

#     a = float(alpha)
#     return ((1.0 - a) * s + a * s_dual).clamp(0.0, 1.0)


# # =====================
# # Load config
# # =====================
# config = get_config()
# print_args(config)

# os.makedirs("./tmp", exist_ok=True)
# os.makedirs("./weights", exist_ok=True)
# os.makedirs("./debug_preds", exist_ok=True)

# report_params(
#     model_name=config.model_name,
#     Scale=config.Scale,
#     Softmax_axis=config.Softmax_axis,
#     Balance=config.Balance,
#     Positional_encoding=config.Positional_encoding,
#     Positional_encoding_shape=config.Positional_encoding_shape,
#     Positional_encoding_way=config.Positional_encoding_way,
#     Dropout_on=config.Dropout_on,
#     Dropout_ratio=config.Dropout_ratio,
#     Classifier_on=config.Classifier_on,
#     CLS_on=config.CLS_on,
#     CLS_mix=config.CLS_mix,
#     key_value_emb=config.key_value_emb,
#     Skip_connection=config.Skip_connection,
#     Layernorm=config.Layernorm
# )

# if not hasattr(config, "datasets"):
#     config.datasets = ["SumMe", "TVSum"]


# # =====================
# # Training
# # =====================
# for dataset in tqdm(config.datasets, total=len(config.datasets), ncols=70, leave=True, position=0):
#     os.makedirs(f"./weights/{dataset}", exist_ok=True)

#     user_scores = get_gt(dataset)

#     if dataset == 'SumMe':
#         batch_size = 1 if config.batch_size == '1' else int(config.SumMe_len * 0.8 * float(config.batch_size))
#     elif dataset == 'TVSum':
#         batch_size = 1 if config.batch_size == '1' else int(config.TVSum_len * 0.8 * float(config.batch_size))
#     else:
#         raise ValueError(f"Unknown dataset: {dataset}")

#     for split_id, (train_loader, test_loader) in tqdm(
#         enumerate(create_dataloader(dataset)),
#         total=5, ncols=70, leave=False, position=1, desc=dataset
#     ):
#         tmp_ckpt = "./tmp/weight.pt"
#         if os.path.exists(tmp_ckpt):
#             try:
#                 os.remove(tmp_ckpt)
#             except Exception:
#                 pass

#         model = set_model(
#             model_name=config.model_name,
#             Scale=config.Scale,
#             Softmax_axis=config.Softmax_axis,
#             Balance=config.Balance,
#             Positional_encoding=config.Positional_encoding,
#             Positional_encoding_shape=config.Positional_encoding_shape,
#             Positional_encoding_way=config.Positional_encoding_way,
#             Dropout_on=config.Dropout_on,
#             Dropout_ratio=config.Dropout_ratio,
#             Classifier_on=config.Classifier_on,
#             CLS_on=config.CLS_on,
#             CLS_mix=config.CLS_mix,
#             key_value_emb=config.key_value_emb,
#             Skip_connection=config.Skip_connection,
#             Layernorm=config.Layernorm
#         ).to(config.device)

#         criterion = torch.nn.MSELoss()
#         optimizer = torch.optim.Adam(
#             model.parameters(),
#             lr=float(config.learning_rate),
#             weight_decay=float(config.weight_decay)
#         )

#         # Low-risk LR schedule (cosine). Keeps structure; safe to disable by setting epochs=1.
#         scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
#             optimizer, T_max=max(1, int(config.epochs)), eta_min=float(config.learning_rate) * 0.1
#         )

#         best_spear = -1e9
#         best_kendall = -1e9
#         saved_samples = 0

#         for epoch in tqdm(
#             range(config.epochs),
#             total=config.epochs, ncols=70, leave=False, position=2,
#             desc=f"Split{split_id+1}"
#         ):
#             # ------------------ TRAIN ------------------
#             model.train()
#             update_loss = None
#             batch = 0

#             for feature, gtscore, dataset_name, video_num in tqdm(
#                 train_loader, ncols=70, leave=False, position=3, desc=f"Epoch{epoch+1}_TRAIN"
#             ):
#                 dataset_name = normalize_dataset_name(dataset_name)
#                 video_num = normalize_video_num(video_num)

#                 feature = feature.to(config.device)
#                 gtscore = gtscore.to(config.device).view(-1)

#                 output = model(feature).view(-1)  # [T] in [0,1] (FSGM uses sigmoid)

#                 if DEBUG:
#                     if not torch.isfinite(output).all():
#                         debug_print("NON-FINITE OUTPUT (TRAIN)", {
#                             "dataset": dataset,
#                             "split": split_id + 1,
#                             "epoch": epoch + 1,
#                             "video_num": video_num,
#                             "output": tensor_stats(output, "output"),
#                             "gtscore": tensor_stats(gtscore, "gtscore"),
#                         })
#                         if DEBUG_FAIL_FAST:
#                             raise FloatingPointError("NaN/Inf detected in model output during TRAIN")

#                     if output.shape != gtscore.shape:
#                         debug_print("SHAPE MISMATCH (TRAIN)", {
#                             "video_num": video_num,
#                             "output_shape": list(output.shape),
#                             "gtscore_shape": list(gtscore.shape),
#                         })
#                         if DEBUG_FAIL_FAST:
#                             raise ValueError("Output and GT shapes differ during TRAIN")

#                 # -----------------------------
#                 # Multi-annotator target (SumMe ONLY)
#                 # -----------------------------
#                 target = gtscore
#                 ann = None

#                 if MULTI_ANNO_ON and (not MULTI_ANNO_SUMME_ONLY or dataset_name == "SumMe"):
#                     if dataset_name == "SumMe":
#                         ann = sample_summe_user_summary_target(video_num, config.device)
#                         ann = _align_1d_tensor_to(ann, int(output.numel()))

#                         if ANNO_MIX_ON:
#                             target = (1.0 - ANNO_MIX_ALPHA) * gtscore + ANNO_MIX_ALPHA * ann
#                         else:
#                             target = ann

#                 # Base regression loss
#                 loss = criterion(output, target)

#                 # Rank regularizer
#                 if RANK_LOSS_ON:
#                     loss_rank = pairwise_rank_loss(output, target, num_pairs=RANK_NUM_PAIRS)
#                     loss = loss + (LAMBDA_RANK * loss_rank)

#                 # Shot smoothness (SumMe only)
#                 if SHOT_SMOOTH_ON and dataset_name == "SumMe":
#                     if video_num not in _summe_meta_cache or "cp" not in _summe_meta_cache[video_num]:
#                         with h5py.File("./data/eccv16_dataset_summe_google_pool5.h5", "r") as hdf:
#                             cp = np.array(hdf[f"{video_num}/change_points"], dtype=np.int64)
#                             pk = np.array(hdf[f"{video_num}/picks"], dtype=np.int64)
#                         if video_num not in _summe_meta_cache:
#                             _summe_meta_cache[video_num] = {"cp": cp, "pk": pk}
#                         else:
#                             _summe_meta_cache[video_num]["cp"] = cp
#                             _summe_meta_cache[video_num]["pk"] = pk

#                     cp = _summe_meta_cache[video_num]["cp"]
#                     pk = _summe_meta_cache[video_num]["pk"]
#                     loss_smooth = summe_shot_smooth_loss(output, cp, pk)
#                     loss = loss + (LAMBDA_SHOT_SMOOTH * loss_smooth)

#                 # Summary-aligned auxiliary (SumMe only, after warmup)
#                 if SUMMARY_ALIGN_ON and (not SUMMARY_ALIGN_SUMME_ONLY or dataset_name == "SumMe"):
#                     if dataset_name == "SumMe" and epoch >= SUMMARY_ALIGN_WARMUP_EPOCHS:
#                         loss_shot = summe_shotwise_mse_aux(output, video_num, device=config.device)
#                         loss = loss + (ALPHA_SHOTWISE_MSE * loss_shot)

#                 # -------------------------------------------------
#                 # Dual-Softmax + Residual Skip auxiliary (SumMe only)
#                 # -------------------------------------------------
#                 if DUAL_SOFTMAX_ON and (not DUAL_SOFTMAX_SUMME_ONLY or dataset_name == "SumMe"):
#                     if dataset_name == "SumMe" and epoch >= DUAL_SOFTMAX_WARMUP_EPOCHS:
#                         out_dual = dual_softmax_residual_mix(
#                             output,
#                             tau_sharp=TAU_SHARP,
#                             tau_smooth=TAU_SMOOTH,
#                             alpha=DUAL_SKIP_ALPHA,
#                         )
#                         loss_dual = criterion(out_dual, target)
#                         loss = loss + (LAMBDA_DUAL * loss_dual)

#                 if DEBUG and epoch == 0 and batch == 0:
#                     dbg = {
#                         "base_mse": float(criterion(output, target).detach().cpu()),
#                         "target_stats": tensor_stats(target, "target"),
#                         "output_stats": tensor_stats(output, "output"),
#                     }
#                     if ann is not None:
#                         dbg["anno_mix_alpha"] = float(ANNO_MIX_ALPHA) if ANNO_MIX_ON else 1.0
#                         dbg["ann_stats"] = {
#                             "min": float(ann.min().item()),
#                             "max": float(ann.max().item()),
#                             "mean": float(ann.mean().item()),
#                         }
#                     if RANK_LOSS_ON:
#                         dbg["rank"] = float(loss_rank.detach().cpu())
#                     if SHOT_SMOOTH_ON and dataset_name == "SumMe":
#                         dbg["shot"] = float(loss_smooth.detach().cpu())
#                     if SUMMARY_ALIGN_ON and dataset_name == "SumMe" and epoch >= SUMMARY_ALIGN_WARMUP_EPOCHS:
#                         dbg["shotwise_aux"] = float(loss_shot.detach().cpu())
#                     if DUAL_SOFTMAX_ON and dataset_name == "SumMe" and epoch >= DUAL_SOFTMAX_WARMUP_EPOCHS:
#                         dbg["dual_aux"] = float(loss_dual.detach().cpu())
#                         dbg["dual_params"] = {
#                             "tau_sharp": TAU_SHARP,
#                             "tau_smooth": TAU_SMOOTH,
#                             "alpha": DUAL_SKIP_ALPHA,
#                             "lambda": LAMBDA_DUAL,
#                         }
#                         dbg["out_dual_stats"] = tensor_stats(out_dual, "out_dual")
#                     debug_print("LOSS COMPONENTS (first batch)", dbg)

#                 # Accumulate micro-batch
#                 update_loss = loss if update_loss is None else (update_loss + loss)
#                 batch += 1

#                 if batch == batch_size:
#                     optimizer.zero_grad()
#                     (update_loss / batch).backward()
#                     optimizer.step()
#                     update_loss = None
#                     batch = 0

#             # leftover
#             if batch > 0 and update_loss is not None:
#                 optimizer.zero_grad()
#                 (update_loss / batch).backward()
#                 optimizer.step()

#             # LR schedule step (per-epoch)
#             try:
#                 scheduler.step()
#             except Exception:
#                 pass

#             # ------------------ VALIDATE ------------------
#             val_spears, val_kendalls = [], []
#             model.eval()

#             with torch.no_grad():
#                 for feature, gtscore, dataset_name, video_num in tqdm(
#                     test_loader, ncols=70, leave=False, position=3, desc=f"Epoch{epoch+1}_TEST"
#                 ):
#                     dataset_name = normalize_dataset_name(dataset_name)
#                     video_num = normalize_video_num(video_num)

#                     feature = feature.to(config.device)
#                     output = model(feature).view(-1)

#                     if DEBUG and (not torch.isfinite(output).all()):
#                         debug_print("NON-FINITE OUTPUT (VAL)", {
#                             "dataset": dataset,
#                             "split": split_id + 1,
#                             "epoch": epoch + 1,
#                             "video_num": video_num,
#                             "output": tensor_stats(output, "output"),
#                         })
#                         if DEBUG_FAIL_FAST:
#                             raise FloatingPointError("NaN/Inf detected in model output during VAL")

#                     if dataset_name in ['SumMe', 'TVSum']:
#                         h5_path = f'./data/eccv16_dataset_{dataset_name.lower()}_google_pool5.h5'
#                         with h5py.File(h5_path, 'r') as hdf:
#                             user_summary = np.array(hdf[f"{video_num}/user_summary"])
#                             sb = np.array(hdf[f"{video_num}/change_points"])
#                             n_frames = int(np.array(hdf[f"{video_num}/n_frames"]))
#                             positions = np.array(hdf[f"{video_num}/picks"])

#                         scores = output.detach().cpu().numpy().tolist()

#                         # SumMe: evaluate on binary summary produced by generate_summary
#                         if dataset_name == 'SumMe':
#                             summary = generate_summary([sb], [scores], [n_frames], [positions])[0]
#                             summary = np.asarray(summary)
#                             spear, kendall = get_corr_coeff([summary], [video_num], dataset_name, user_summary)
#                         else:
#                             # TVSum: correlate predicted scores with ground-truth user scores
#                             spear, kendall = get_corr_coeff([scores], [video_num], dataset_name, user_scores)

#                         val_spears.append(spear)
#                         val_kendalls.append(kendall)

#                         if DEBUG and DEBUG_SAVE_SAMPLES and saved_samples < DEBUG_SAVE_MAX:
#                             dbg_obj = {
#                                 "dataset": dataset,
#                                 "split": int(split_id + 1),
#                                 "epoch": int(epoch + 1),
#                                 "video_num": video_num,
#                                 "scores_min": float(np.min(scores)) if len(scores) else None,
#                                 "scores_max": float(np.max(scores)) if len(scores) else None,
#                                 "scores_mean": float(np.mean(scores)) if len(scores) else None,
#                                 "scores_head": scores[:20],
#                                 "scores_tail": scores[-20:],
#                                 "picks_len": int(len(positions)),
#                                 "n_frames": int(n_frames),
#                             }
#                             out_path = f"./debug_preds/{dataset}/split{split_id+1}/epoch{epoch+1}_{video_num.replace('/', '_')}.json"
#                             save_json(out_path, dbg_obj)
#                             saved_samples += 1

#             mk = safe_nanmean(val_kendalls)
#             ms = safe_nanmean(val_spears)

#             if DEBUG and ((epoch + 1) % DEBUG_EVERY_N_EPOCHS == 0):
#                 debug_print("EPOCH VAL SUMMARY", {
#                     "dataset": dataset,
#                     "split": split_id + 1,
#                     "epoch": epoch + 1,
#                     "val_count": int(len(val_spears)),
#                     "mean_spear": ms,
#                     "mean_kendall": mk,
#                     "best_spear": float(best_spear),
#                     "best_kendall": float(best_kendall),
#                 })

#             # Save best: PRIORITIZE Kendall (matches your reporting + ranking objective)
#             if np.isfinite(ms) and np.isfinite(mk):
#                 improved = (mk > best_kendall) or (mk == best_kendall and ms > best_spear)
#                 if improved:
#                     best_spear = float(ms)
#                     best_kendall = float(mk)
#                     torch.save(model.state_dict(), tmp_ckpt)

#         # Guarantee checkpoint exists
#         if not os.path.exists(tmp_ckpt):
#             if DEBUG:
#                 debug_print("NO BEST CKPT SAVED; FALLING BACK TO FINAL", {
#                     "dataset": dataset,
#                     "split": split_id + 1,
#                     "best_spear": float(best_spear),
#                     "best_kendall": float(best_kendall),
#                 })
#             torch.save(model.state_dict(), tmp_ckpt)

#         dst_ckpt = f'./weights/{dataset}/split{split_id+1}.pt'
#         shutil.move(tmp_ckpt, dst_ckpt)




























