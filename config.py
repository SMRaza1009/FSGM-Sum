import argparse
import numpy as np
import random
import torch

# Process bool argument
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError(f"Boolean value expected, got {v}")

# Process none argument
def str2none(v):
    if v.lower()=='none':
        return None
    else:
        return v

# Define configuration class
class Config(object):
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        for k, v in kwargs.items():
            setattr(self, k, v)

        self.SumMe_len = 25
        self.TVSum_len = 50

        # Set device
        if self.device!='cpu':
            torch.cuda.set_device(self.device)

        # Set seed
        self.set_seed()

    # Set the seed
    def set_seed(self):
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        if self.device!='cpu':
            torch.cuda.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

# Define all configurations
def get_config(parse=True, **optional_kwargs):
    parser = argparse.ArgumentParser()

    # Run settings
    parser.add_argument('--seed', type=int, default=123456)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--datasets', type=str, nargs='+', default=['SumMe', 'TVSum'], choices=['SumMe', 'TVSum'])
    parser.add_argument('--weights_dir', type=str, default='./weights',
                        help='Checkpoints are written to / read from <weights_dir>/<dataset>/split<k>.pt')
    parser.add_argument('--results_dir', type=str, default='./results')

    # Evaluation protocol
    parser.add_argument('--split_dir', type=str, default='./splits',
                        help='Directory containing <dataset>_splits.txt')
    parser.add_argument('--val_ratio', type=float, default=0.125,
                        help='Fraction of each split\'s training videos held out for checkpoint selection '
                             '(0.125 gives a 7:1:2 train/val/test ratio). Test videos are never used for selection.')

    # Optimization
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', default='1')
    parser.add_argument('--learning_rate', default='1e-3')
    parser.add_argument('--weight_decay', default='1e-7')

    # Model
    parser.add_argument('--model_name', type=str, default='FSGM_Attention', choices=['FSGM_Attention', 'FSGM'])
    parser.add_argument('--Scale', type=str2none, default=None)
    parser.add_argument('--Softmax_axis', type=str2none, default='TD')
    parser.add_argument('--Balance', type=str2none, default=None)

    parser.add_argument('--Positional_encoding', type=str2none, default='FPE')
    parser.add_argument('--Positional_encoding_shape', type=str2none, default='TD')
    parser.add_argument('--Positional_encoding_way', type=str2none, default='PGL_SUM')
    parser.add_argument('--Dropout_on', type=str2bool, default=True)
    parser.add_argument('--Dropout_ratio', default='0.6', help='Dropout on the calibrated attention maps')

    parser.add_argument('--Classifier_on', type=str2bool, default=True)
    parser.add_argument('--CLS_on', type=str2bool, default=True)
    parser.add_argument('--CLS_mix', type=str2none, default='Final')

    parser.add_argument('--key_value_emb', type=str2none, default='kv')
    parser.add_argument('--Skip_connection', type=str2none, default='KC')
    parser.add_argument('--Layernorm', type=str2bool, default=True)

    # Factorized gate (defaults = full model; other values reproduce the architecture ablations)
    parser.add_argument('--fsgm_hidden', type=int, default=64)
    parser.add_argument('--fsgm_dilations', type=int, nargs='+', default=[1, 2, 4])
    parser.add_argument('--fsgm_fft_bias', type=str, default='timestep', choices=['timestep', 'scalar', 'none'])
    parser.add_argument('--fsgm_branches', type=str, default='both', choices=['both', 'temporal', 'channel', 'none'])
    parser.add_argument('--fsgm_pooling', type=str, default='meanmax', choices=['meanmax', 'mean', 'max'])
    parser.add_argument('--fsgm_fusion', type=str, default='add', choices=['add', 'mul'])
    parser.add_argument('--temp_learnable', type=str2bool, default=True)

    # Training objective (defaults = full objective; disable terms for the loss ablations)
    parser.add_argument('--multi_anno', type=str2bool, default=True, help='Mix a sampled user summary into the SumMe target')
    parser.add_argument('--anno_mix_alpha', type=float, default=0.5)
    parser.add_argument('--rank_loss', type=str2bool, default=True)
    parser.add_argument('--rank_num_pairs', type=int, default=1024)
    parser.add_argument('--lambda_rank', type=float, default=0.10)
    parser.add_argument('--shot_smooth_loss', type=str2bool, default=True)
    parser.add_argument('--lambda_shot_smooth', type=float, default=0.06)
    parser.add_argument('--shotwise_loss', type=str2bool, default=True)
    parser.add_argument('--alpha_shotwise', type=float, default=0.08)
    parser.add_argument('--dual_softmax_loss', type=str2bool, default=True)
    parser.add_argument('--lambda_dual', type=float, default=0.08)
    parser.add_argument('--tau_sharp', type=float, default=0.40)
    parser.add_argument('--tau_smooth', type=float, default=1.50)
    parser.add_argument('--dual_skip_alpha', type=float, default=0.15)
    parser.add_argument('--aux_warmup_epochs', type=int, default=5,
                        help='Epochs before the shotwise and dual-softmax terms are activated')

    # Generate summary videos
    parser.add_argument('--input_is_file', type=str2bool, default='true')
    parser.add_argument('--file_path', type=str, default='./videos/video.mp4')
    parser.add_argument('--dir_path', type=str, default='./videos')
    parser.add_argument('--ext', type=str, default='mp4')
    parser.add_argument('--sample_rate', type=int, default=15)
    parser.add_argument('--save_path', type=str, default='./summary_videos')
    parser.add_argument('--weight_path', type=str, default='./weights/SumMe/split1.pt')

    kwargs = vars(parser.parse_args())
    kwargs.update(optional_kwargs)

    return Config(**kwargs)
