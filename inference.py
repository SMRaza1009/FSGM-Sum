import json
import os
import numpy as np
import torch

from config import get_config
from dataset import create_dataloader
from evaluation_metrics import evaluate
from model import build_model
from utils import report_params,get_gt

# Load configurations
config = get_config()

# Print the number of parameters
report_params(build_model(config))

# Evaluate the released checkpoints on the test videos of every split
all_results = {}
for dataset in config.datasets:
    user_scores = get_gt(dataset)
    rows = []

    for split_id,(_,_,test_loader) in enumerate(create_dataloader(dataset, split_dir=config.split_dir, val_ratio=0)):
        model = build_model(config)
        model.load_state_dict(torch.load(f'{config.weights_dir}/{dataset}/split{split_id+1}.pt', map_location='cpu'))
        model.to(config.device)

        result = evaluate(model, test_loader, dataset, user_scores, config.device)
        rows.append(result)
        print("[{} Split{}] Kendall: {:.3f}, Spearman: {:.3f}, F1: {:.3f}".format(
            dataset, split_id+1, result['kendall'], result['spearman'], result['f1']))

    summary = {'splits': rows}
    for key in ['kendall', 'spearman', 'f1']:
        values = np.array([r[key] for r in rows], dtype=np.float64)
        summary[key + '_mean'] = float(values.mean())
        summary[key + '_std'] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
    all_results[dataset] = summary

    print("[FINAL - {}] Kendall: {:.3f} ± {:.3f}, Spearman: {:.3f} ± {:.3f}, F1: {:.3f} ± {:.3f} (mean ± sample std)".format(
        dataset, summary['kendall_mean'], summary['kendall_std'], summary['spearman_mean'],
        summary['spearman_std'], summary['f1_mean'], summary['f1_std']))
    print()

os.makedirs(config.results_dir, exist_ok=True)
with open(f'{config.results_dir}/inference_results.json', 'w') as f:
    json.dump(all_results, f, indent=2)
