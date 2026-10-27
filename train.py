import os
import json
import h5py
import numpy as np
import shutil
import torch
import torch.nn.functional as F

from tqdm import tqdm

from config import get_config
from dataset import create_dataloader, h5_path
from evaluation_metrics import evaluate
from model import build_model
from utils import report_params, print_args, get_gt

# Cache SumMe metadata (change points, picks, user summaries) per video
_summe_meta_cache = {}


def normalize_video_num(video_num):
    """Make video_num a clean string key for H5 access."""
    if isinstance(video_num, (list, tuple)) and len(video_num) == 1:
        video_num = video_num[0]
    if isinstance(video_num, np.ndarray) and video_num.size == 1:
        video_num = video_num.item()
    if isinstance(video_num, bytes):
        video_num = video_num.decode("utf-8")
    return str(video_num)


def summe_meta(video_num):
    if video_num not in _summe_meta_cache:
        with h5py.File(h5_path('SumMe'), "r") as hdf:
            _summe_meta_cache[video_num] = {
                "cp": np.array(hdf[f"{video_num}/change_points"], dtype=np.int64),
                "pk": np.array(hdf[f"{video_num}/picks"], dtype=np.int64),
                "user_summary": np.array(hdf[f"{video_num}/user_summary"], dtype=np.float32),
            }
    return _summe_meta_cache[video_num]


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

    us = summe_meta(video_num)["user_summary"]
    pk = summe_meta(video_num)["pk"]

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

    cp = summe_meta(video_num)["cp"]
    pk = summe_meta(video_num)["pk"]
    us = summe_meta(video_num)["user_summary"]

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
# Training objective (Eq. total loss; SumMe-only terms as in Sec. 3.5)
# =====================
def compute_loss(output, gtscore, dataset_name, video_num, epoch, config, criterion):
    # Multi-annotator target (SumMe only)
    target = gtscore
    if config.multi_anno and dataset_name == "SumMe":
        ann = sample_summe_user_summary_target(video_num, config.device)
        ann = _align_1d_tensor_to(ann, int(output.numel()))
        target = (1.0 - config.anno_mix_alpha) * gtscore + config.anno_mix_alpha * ann

    loss = criterion(output, target)

    if config.rank_loss:
        loss = loss + config.lambda_rank * pairwise_rank_loss(output, target, num_pairs=config.rank_num_pairs)

    if dataset_name == "SumMe":
        if config.shot_smooth_loss:
            meta = summe_meta(video_num)
            loss = loss + config.lambda_shot_smooth * summe_shot_smooth_loss(output, meta["cp"], meta["pk"])

        if epoch >= config.aux_warmup_epochs:
            if config.shotwise_loss:
                loss = loss + config.alpha_shotwise * summe_shotwise_mse_aux(output, video_num, device=config.device)
            if config.dual_softmax_loss:
                out_dual = dual_softmax_residual_mix(
                    output,
                    tau_sharp=config.tau_sharp,
                    tau_smooth=config.tau_smooth,
                    alpha=config.dual_skip_alpha,
                )
                loss = loss + config.lambda_dual * criterion(out_dual, target)

    return loss


def train_one_epoch(model, optimizer, criterion, train_loader, batch_size, epoch, config):
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

        output = model(feature).view(-1)  # [T] in [0,1]
        if not torch.isfinite(output).all():
            raise FloatingPointError(f"NaN/Inf in model output for {dataset_name}/{video_num} (epoch {epoch+1})")
        if output.shape != gtscore.shape:
            raise ValueError(f"Output {tuple(output.shape)} and GT {tuple(gtscore.shape)} shapes differ for {video_num}")

        loss = compute_loss(output, gtscore, dataset_name, video_num, epoch, config, criterion)

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


def main():
    config = get_config()
    print_args(config)
    report_params(build_model(config))

    os.makedirs("./tmp", exist_ok=True)
    os.makedirs(config.results_dir, exist_ok=True)
    all_results = {}

    for dataset in tqdm(config.datasets, total=len(config.datasets), ncols=70, leave=True, position=0):
        os.makedirs(f"{config.weights_dir}/{dataset}", exist_ok=True)
        user_scores = get_gt(dataset)

        if dataset == 'SumMe':
            batch_size = 1 if config.batch_size == '1' else int(config.SumMe_len * 0.8 * float(config.batch_size))
        else:
            batch_size = 1 if config.batch_size == '1' else int(config.TVSum_len * 0.8 * float(config.batch_size))

        split_results = []
        loaders = create_dataloader(dataset, split_dir=config.split_dir, val_ratio=config.val_ratio, seed=config.seed)
        for split_id, (train_loader, val_loader, test_loader) in tqdm(
            enumerate(loaders), total=len(loaders), ncols=70, leave=False, position=1, desc=dataset
        ):
            tmp_ckpt = "./tmp/weight.pt"
            if os.path.exists(tmp_ckpt):
                os.remove(tmp_ckpt)

            model = build_model(config).to(config.device)
            criterion = torch.nn.MSELoss()
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=float(config.learning_rate),
                weight_decay=float(config.weight_decay)
            )

            best = {'spearman': -np.inf, 'kendall': -np.inf, 'epoch': None}
            for epoch in tqdm(range(config.epochs), total=config.epochs, ncols=70, leave=False, position=2,
                              desc=f"Split{split_id+1}"):
                train_one_epoch(model, optimizer, criterion, train_loader, batch_size, epoch, config)

                # Checkpoint selection on the held-out validation videos only:
                # highest Spearman's rho, Kendall's tau as tie-breaker.
                val = evaluate(model, val_loader, dataset, user_scores, config.device)
                ms, mk = val['spearman'], val['kendall']
                if np.isfinite(ms) and np.isfinite(mk):
                    if ms > best['spearman'] or (ms == best['spearman'] and mk > best['kendall']):
                        best = {'spearman': ms, 'kendall': mk, 'epoch': epoch + 1}
                        torch.save(model.state_dict(), tmp_ckpt)

            if not os.path.exists(tmp_ckpt):
                print(f"[{dataset} split{split_id+1}] no finite validation score; keeping the final epoch")
                best['epoch'] = config.epochs
                torch.save(model.state_dict(), tmp_ckpt)

            dst_ckpt = f'{config.weights_dir}/{dataset}/split{split_id+1}.pt'
            shutil.move(tmp_ckpt, dst_ckpt)

            # Test videos are evaluated once, with the selected checkpoint
            model.load_state_dict(torch.load(dst_ckpt, map_location='cpu'))
            test = evaluate(model, test_loader, dataset, user_scores, config.device)
            split_results.append({
                'split': split_id + 1,
                'selected_epoch': best['epoch'],
                'val_spearman': best['spearman'],
                'val_kendall': best['kendall'],
                'test_kendall': test['kendall'],
                'test_spearman': test['spearman'],
                'test_f1': test['f1'],
            })
            tqdm.write("[{} Split{}] epoch {} | val rho {:.3f} | test tau {:.3f}, rho {:.3f}, F1 {:.3f}".format(
                dataset, split_id + 1, best['epoch'], best['spearman'], test['kendall'], test['spearman'], test['f1']))

        all_results[dataset] = summarize(dataset, split_results)

    with open(f"{config.results_dir}/train_results.json", "w") as f:
        json.dump({'config': config.kwargs, 'results': all_results}, f, indent=2)
    print(f"Saved {config.results_dir}/train_results.json")


def summarize(dataset, split_results):
    out = {'splits': split_results}
    for key in ['test_kendall', 'test_spearman', 'test_f1']:
        values = np.array([r[key] for r in split_results], dtype=np.float64)
        out[key + '_mean'] = float(values.mean())
        out[key + '_std'] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    print("[FINAL - {}] Kendall: {:.3f} ± {:.3f}, Spearman: {:.3f} ± {:.3f}, F1: {:.3f} ± {:.3f} (mean ± sample std)".format(
        dataset, out['test_kendall_mean'], out['test_kendall_std'], out['test_spearman_mean'],
        out['test_spearman_std'], out['test_f1_mean'], out['test_f1_std']))
    return out


if __name__ == '__main__':
    main()
