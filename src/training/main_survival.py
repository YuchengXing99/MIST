"""
Main entry point for survival downstream tasks
"""

from __future__ import print_function

import argparse
import pdb
import os
from os.path import join as j_
import sys

# internal imports
from utils.file_utils import save_pkl
import glob as glob_module

from utils.utils import (seed_torch, array2list, merge_dict, read_splits,
                         parse_model_name, extract_patching_info)

from .trainer import train
from wsi_datasets import WSISurvivalDataset
# pytorch imports
import torch
from torch.utils.data import DataLoader

import pandas as pd
import json

def compute_fold_summary(seed_dir):
    """Aggregate per-fold summary.csv files into all_folds.csv + summary.csv with mean/std."""
    fold_dirs = sorted(glob_module.glob(j_(seed_dir, 'fold_*')))
    rows = []
    for fd in fold_dirs:
        summary_path = j_(fd, 'summary.csv')
        if not os.path.isfile(summary_path):
            continue
        df = pd.read_csv(summary_path)
        fold_k = int(os.path.basename(fd).split('_')[1])
        df.insert(0, 'fold', fold_k)
        rows.append(df)
    if not rows:
        return
    all_folds_df = pd.concat(rows, ignore_index=True)
    all_folds_df.to_csv(j_(seed_dir, 'all_folds.csv'), index=False)

    numeric_cols = [c for c in all_folds_df.select_dtypes(include='number').columns if c != 'fold']
    mean_row = all_folds_df[numeric_cols].mean().to_frame().T
    std_row  = all_folds_df[numeric_cols].std().to_frame().T
    mean_row.insert(0, 'fold', 'mean')
    std_row.insert(0, 'fold', 'std')
    summary = pd.concat([all_folds_df[['fold'] + numeric_cols], mean_row, std_row], ignore_index=True)
    summary.to_csv(j_(seed_dir, 'summary.csv'), index=False)
    print(f"[Summary] Written to {j_(seed_dir, 'summary.csv')}")


def build_datasets(csv_splits, model_type, batch_size=1, num_workers=2, train_kwargs={}, val_kwargs={}):
    """
    Construct dataloaders from the data splits
    """
    dataset_splits = {}
    label_bins = None
    for k in csv_splits.keys(): # ['train', 'val', 'test']
        df = csv_splits[k]
        dataset_kwargs = train_kwargs.copy() if (k == 'train') else val_kwargs.copy()
        dataset_kwargs['label_bins'] = label_bins
        dataset = WSISurvivalDataset(df, **dataset_kwargs)

        batch_size = batch_size if dataset_kwargs.get('bag_size', -1) > 0 else 1

        dataloader = DataLoader(dataset, batch_size=batch_size,
                                shuffle=dataset_kwargs['shuffle'], num_workers=num_workers)
        dataset_splits[k] = dataloader
        print(f'split: {k}, n: {len(dataset)}')
        if (args.loss_fn == 'nll') and (k == 'train'):
            label_bins = dataset.get_label_bins()
    return dataset_splits


def main(args):
    if args.train_bag_size == -1:
        args.train_bag_size = args.bag_size
    if args.val_bag_size == -1:
        args.val_bag_size = args.bag_size
    if args.loss_fn != 'nll':
        args.n_label_bins = 0

    censorship_col = args.target_col.split('_')[0] + '_censorship'

    train_kwargs = dict(data_source=args.data_source,
                        survival_time_col=args.target_col,
                        censorship_col=censorship_col,
                        n_label_bins=args.n_label_bins,
                        label_bins=None,
                        bag_size=args.train_bag_size,
                        shuffle=True,
                        )

    # use the whole bag at test time
    val_kwargs = dict(data_source=args.data_source,
                      survival_time_col=args.target_col,
                      censorship_col=censorship_col,
                      n_label_bins=args.n_label_bins,
                      label_bins=None,
                      bag_size=args.val_bag_size,
                      shuffle=False,
                      )

    all_results, all_dumps = {}, {}

    # Cross-validation
    seed_torch(args.seed)
    csv_splits = read_splits(args)
    print('successfully read splits for: ', list(csv_splits.keys()))
    dataset_splits = build_datasets(csv_splits, 
                                    model_type=args.model_type,
                                    batch_size=args.batch_size,
                                    num_workers=args.num_workers,
                                    train_kwargs=train_kwargs,
                                    val_kwargs=val_kwargs)

    fold_results, fold_dumps = train(dataset_splits, args, mode='survival')

    for split, split_results in fold_results.items():
        all_results[split] = merge_dict({}, split_results) if (split not in all_results.keys()) else merge_dict(all_results[split], split_results)
        save_pkl(j_(args.results_dir, f'{split}_results.pkl'), fold_dumps[split]) # saves per-split, per-fold results to pkl
    
    final_dict = {}
    for split, split_results in all_results.items():
        final_dict.update({f'{metric}_{split}': array2list(val) for metric, val in split_results.items()})
    final_df = pd.DataFrame(final_dict)
    save_name = 'summary.csv'
    final_df.to_csv(j_(args.results_dir, save_name), index=False)
    with open(j_(args.results_dir, save_name + '.json'), 'w') as f:
        f.write(json.dumps(final_dict, sort_keys=True, indent=4))
    
    dump_path = j_(args.results_dir, 'all_dumps.h5')
    save_pkl(dump_path, fold_dumps)

    return final_dict

# Generic training settings
parser = argparse.ArgumentParser(description='Configurations for WSI Training')
### optimizer settings ###
parser.add_argument('--max_epochs', type=int, default=20,
                    help='maximum number of epochs to train (default: 20)')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='learning rate')
parser.add_argument('--wd', type=float, default=1e-5,
                    help='weight decay')
parser.add_argument('--accum_steps', type=int, default=1,
                    help='grad accumulation steps')
parser.add_argument('--opt', type=str, default='adamW',
                    choices=['adamW', 'sgd', 'RAdam'])
parser.add_argument('--lr_scheduler', type=str,
                    choices=['cosine', 'linear', 'constant'], default='constant')
parser.add_argument('--warmup_steps', type=int,
                    default=-1, help='warmup iterations')
parser.add_argument('--warmup_epochs', type=int,
                    default=-1, help='warmup epochs')
parser.add_argument('--batch_size', type=int, default=1)

### misc ###
parser.add_argument('--print_every', default=100,
                    type=int, help='how often to print')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed for reproducible experiment (default: 1)')
parser.add_argument('--num_workers', type=int, default=2)

### Earlystopper args ###
parser.add_argument('--early_stopping', type=int,
                    default=0, help='enable early stopping')
parser.add_argument('--es_min_epochs', type=int, default=3,
                    help='early stopping min epochs')
parser.add_argument('--es_patience', type=int, default=5,
                    help='early stopping min patience')
parser.add_argument('--es_metric', type=str, default='loss',
                    help='early stopping metric')

### model args ###
parser.add_argument('--model_type', type=str,
                    choices=['ABMIL', 'TransMIL', 'ILRA',
                             'MeanMIL', 'MaxMIL', 'CLAM', 'TransformerMIL', 'DSMIL'],
                    default='ABMIL',
                    help='type of model (use --model_config *_MIST_default to enable MIST)')
parser.add_argument('--emb_model_type', type=str, default='LinEmb_LR')
parser.add_argument('--ot_eps', default=0.1, type=float,
                    help='Strength for entropic constraint regularization for OT')
parser.add_argument('--model_config', type=str,
                    default='ABMIL_default', help="name of model config file")
parser.add_argument('--n_fc_layers', type=int)
parser.add_argument('--em_iter', type=int)
parser.add_argument('--tau', type=float)
parser.add_argument('--out_type', type=str, default='param_cat')

# Prototype related
parser.add_argument('--load_proto', action='store_true', default=False)
parser.add_argument('--proto_path', type=str)
parser.add_argument('--fix_proto', action='store_true', default=False)
parser.add_argument('--n_proto', type=int)

parser.add_argument('--in_dim', default=768, type=int,
                    help='dim of input features')
parser.add_argument('--in_dropout', default=0.1, type=float,
                    help='Probability of dropping out input features.')
parser.add_argument('--bag_size', type=int, default=-1)
parser.add_argument('--train_bag_size', type=int, default=-1)
parser.add_argument('--val_bag_size', type=int, default=-1)
parser.add_argument('--loss_fn', type=str, default='nll', choices=['nll', 'cox', 'sumo', 'ipcwls', 'rank'],
                    help='which loss function to use')
parser.add_argument('--nll_alpha', type=float, default=0,
                    help='Balance between censored / uncensored loss')

# experiment task / label args ###
parser.add_argument('--exp_code', type=str, default=None,
                    help='experiment code for saving results')
parser.add_argument('--task', type=str, default='unspecified_survival_task')
parser.add_argument('--target_col', type=str, default='os_survival_days')
parser.add_argument('--n_label_bins', type=int, default=4,
                    help='number of bins for event time discretization')

# dataset / split args ###
parser.add_argument('--data_source', type=str, default=None,
                    help='manually specify the data source')
parser.add_argument('--split_dir', type=str, default=None,
                    help='manually specify the set of splits to use')
parser.add_argument('--split_names', type=str, default='train,val,test',
                    help='delimited list for specifying names within each split')
parser.add_argument('--overwrite', action='store_true', default=False,
                    help='overwrite existing results')

# logging args ###
parser.add_argument('--results_dir', default='./results',
                    help='results directory (default: ./results)')
parser.add_argument('--tags', nargs='+', type=str, default=None,
                    help='tags for logging')
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":

    print('task: ', args.task)
    args.split_dir = j_('splits', args.split_dir)
    print('split_dir: ', args.split_dir)
    split_num = args.split_dir.split('/')[2].split('_k=')
    args.split_name_clean = args.split_dir.split('/')[2].split('_k=')[0]
    if len(split_num) > 1:
        args.split_k = int(split_num[1])
    else:
        args.split_k = 0

    if args.load_proto:
        assert os.path.isfile(args.proto_path), f"Path {args.proto_path} doesn't exist!"
        args.proto_fname = '/'.join(args.proto_path.split('/')[-2:])

    ### Allows you to pass in multiple data sources (separated by comma). If single data source, no change.
    args.data_source = [src for src in args.data_source.split(',')]
    check_params_same = []
    for src in args.data_source:
        print('data source: ', src)
        assert os.path.isdir(src), f"data source must be a directory: {src} invalid"

        feat_name = os.path.basename(src)
        mag, patch_size = extract_patching_info(os.path.dirname(src))
        if (mag < 0 or patch_size < 0):
            raise ValueError(f"invalid patching info parsed for {src}")
        check_params_same.append([feat_name, mag, patch_size])

        parsed = parse_model_name(feat_name)
        parsed.update({'patch_mag': mag, 'patch_size': patch_size})

    try:
        check_params_same = pd.DataFrame(check_params_same, columns=['feats_name', 'mag', 'patch_size'])
        assert check_params_same.drop(['feats_name'], axis=1).drop_duplicates().shape[0] == 1
        print("All data sources have the same feature extraction parameters.")
    except:
        print("Data sources do not share the same feature extraction parameters. Exiting...")
        sys.exit()

    for key, val in parsed.items():
        setattr(args, key, val)

    ### setup results dir: <results_dir>/<model_name>/seed_<seed>/fold_<k>/ ###
    model_name = args.model_config.lower().replace('_default', '')
    seed_dir = j_(args.results_dir, model_name, f'seed_{args.seed}')
    args.results_dir = j_(seed_dir, f'fold_{args.split_k}')
    os.makedirs(args.results_dir, exist_ok=True)

    print("\n################### Settings ###################")
    for key, val in vars(args).items():
        print("{}:  {}".format(key, val))

    with open(j_(args.results_dir, 'config.json'), 'w') as f:
        f.write(json.dumps(vars(args), sort_keys=True, indent=4))

    #### train ####
    results = main(args)

    ### aggregate fold results if all folds present ###
    compute_fold_summary(seed_dir)

    print("FINISHED!\n\n\n")