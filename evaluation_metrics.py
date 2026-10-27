import h5py
import numpy as np
import torch
from scipy.stats import spearmanr, kendalltau, rankdata

from dataset import h5_path
from generate_summary import generate_summary

# Calculate Kendall's and Spearman's coefficients
def get_corr_coeff(pred_imp_scores, videos, dataset, user_scores=None):
    rho_coeff, tau_coeff = [], []
    if dataset=='SumMe':
        for pred_imp_score,video in zip(pred_imp_scores,videos):
            true = np.mean(user_scores,axis=0)
            rho_coeff.append(spearmanr(pred_imp_score,true)[0])
            tau_coeff.append(kendalltau(rankdata(pred_imp_score),rankdata(true))[0])
    elif dataset=='TVSum':
        for pred_imp_score,video in zip(pred_imp_scores,videos):
            pred_imp_score = np.squeeze(pred_imp_score).tolist()
            user = int(video.split("_")[-1])

            curr_user_score = user_scores[user-1]

            tmp_rho_coeff, tmp_tau_coeff = [], []
            for annotation in range(len(curr_user_score)):
                true_user_score = curr_user_score[annotation]
                curr_rho_coeff, _ = spearmanr(pred_imp_score, true_user_score)
                curr_tau_coeff, _ = kendalltau(rankdata(pred_imp_score), rankdata(true_user_score))
                tmp_rho_coeff.append(curr_rho_coeff)
                tmp_tau_coeff.append(curr_tau_coeff)
            rho_coeff.append(np.mean(tmp_rho_coeff))
            tau_coeff.append(np.mean(tmp_tau_coeff))
    rho_coeff = np.array(rho_coeff).mean()
    tau_coeff = np.array(tau_coeff).mean()
    return rho_coeff, tau_coeff

# Keyshot F1 between a binary summary and every user summary ('max' for SumMe, 'avg' for TVSum)
def get_f1(summary, user_summary, mode='max'):
    summary = np.asarray(summary, dtype=bool)
    f1s = []
    for user in np.asarray(user_summary, dtype=bool):
        n = min(len(summary), len(user))
        pred, gt = summary[:n], user[:n]
        overlap = np.logical_and(pred, gt).sum()
        if overlap == 0:
            f1s.append(0.0)
            continue
        precision = overlap / pred.sum()
        recall = overlap / gt.sum()
        f1s.append(2 * precision * recall / (precision + recall))
    return float(np.max(f1s)) if mode == 'max' else float(np.mean(f1s))

# Evaluate a model on one loader; returns mean Kendall's tau, Spearman's rho, and keyshot F1
def evaluate(model, loader, dataset, user_scores, device):
    model.eval()
    kendalls, spears, f1s = [], [], []
    with torch.no_grad(), h5py.File(h5_path(dataset),'r') as hdf:
        for feature,_,dataset_name,video_num in loader:
            output = model(feature.to(device)).view(-1)
            if not torch.isfinite(output).all():
                raise FloatingPointError(f"NaN/Inf in model output for {dataset}/{video_num}")

            user_summary = np.array(hdf[video_num]['user_summary'])
            sb = np.array(hdf[f"{video_num}/change_points"])
            n_frames = int(np.array(hdf[f"{video_num}/n_frames"]))
            positions = np.array(hdf[f"{video_num}/picks"])
            scores = output.cpu().numpy().tolist()
            summary = generate_summary([sb], [scores], [n_frames], [positions])[0]

            # SumMe correlates the decoded keyshot summary; TVSum correlates the raw scores
            if dataset_name=='SumMe':
                spear,kendall = get_corr_coeff([summary],[video_num],dataset_name,user_summary)
            else:
                spear,kendall = get_corr_coeff([scores],[video_num],dataset_name,user_scores)

            spears.append(spear)
            kendalls.append(kendall)
            f1s.append(get_f1(summary, user_summary, mode='max' if dataset_name=='SumMe' else 'avg'))

    return {
        'kendall': float(np.nanmean(kendalls)),
        'spearman': float(np.nanmean(spears)),
        'f1': float(np.mean(f1s)),
    }
