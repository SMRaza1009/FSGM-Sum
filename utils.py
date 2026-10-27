import csv
import numpy as np
import torch

from collections import Counter

# Count trainable parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Print the number of parameters (millions) and the size of the 32-bit weights (MB), as in Table 4
def report_params(model):
    n = count_parameters(model)
    print(f"PARAMS: {n / 1e6:.2f}M ({n * 4 / 2**20:.2f} MB)")

# Print all arguments and GPU setting
def print_args(args):
    print(args.kwargs)
    print(f"CUDA: {torch.version.cuda}")
    print(f"cuDNN: {torch.backends.cudnn.version()}")
    if 'cuda' in args.device:
        print(f"GPU: {torch.cuda.is_available()}")
        print(f"GPU count: {torch.cuda.device_count()}")
        print(f"GPU name: {torch.cuda.get_device_name(0)}")

# Load ground truth for TVSum
def get_gt(dataset):
    if dataset=='TVSum':
        annot_path = f"./data/ydata-anno.tsv"
        with open(annot_path) as annot_file:
            annot = list(csv.reader(annot_file, delimiter="\t"))
        annotation_length = list(Counter(np.array(annot)[:, 0]).values())
        user_scores = []
        for idx in range(1,51):
            init = (idx - 1) * annotation_length[idx-1]
            till = idx * annotation_length[idx-1]
            user_score = []
            for row in annot[init:till]:
                curr_user_score = row[2].split(",")
                curr_user_score = np.array([float(num) for num in curr_user_score])
                curr_user_score = curr_user_score / curr_user_score.max(initial=-1)
                curr_user_score = curr_user_score[::15]

                user_score.append(curr_user_score)
            user_scores.append(user_score)
        return user_scores
    elif dataset=='SumMe':
        return None
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
