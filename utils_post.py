import numpy as np
import torch
from collections import defaultdict, Counter
import pdb
import json
import os
import argparse
from config import *

N_seeds, N_formats = 5, 3
N_exps = N_seeds * N_formats
TASK_LIST = list(Tasks.keys())
MODEL_LIST = [('llama', 7), ('mistral', 7), ('llama3', 8)]

def save_results(fn, RES):
    with open(fn , "w" ) as write:
        json.dump(RES , write)


def model_map(model, size):
    if model == 'llama':
        return f'meta-llama/Llama-2-{size}b-hf'
    elif model == 'llama3':
        return "meta-llama/Meta-Llama-3-8B"
    else:
        return 'mistralai/Mistral-7B-Instruct-v0.1'


def compact(RES, fn):
    # get avg, std over N_exps
    summary = defaultdict(lambda: defaultdict(str))
    for md, task2accs in RES.items():
        task_avg = []
        for task, accs in task2accs.items():
            try:
                assert len(accs) == N_exps
            except:
                print(fn, md, task, len(accs))
            accs = np.array(accs)
            summary[md][task] = f'{100*accs.mean():.1f} ± {100*accs.std():.1f}'
            task_avg.append(accs.mean())
        summary[md]['Task_Avg'] = f'{np.array(task_avg).mean():.1%}'

    save_results(os.path.join('Summary', f'{fn}_flatten.json'), RES)
    save_results(os.path.join('Summary', f'{fn}_mean_std.json'), summary)
    return summary


def load(fn, out_dir='Summary'):
    with open(os.path.join(out_dir, f'{fn}_flatten.json')) as f:
        RES = json.load(f)
    with open(os.path.join(out_dir, f'{fn}_mean_std.json')) as f:
        summary = json.load(f)
    return RES, summary


def get_logits_accs_labels(model_size, task, form_i, seed, return_projs=False):
    model, size = model_size
    n_shots = 3 if task == 'mnli' else 4
    cur_dir = f'DECOMP/{model_map(model, size)}/{task}/{n_shots}shots/f{form_i}'
    #print(cur_dir)

    whole_acc = np.load(os.path.join(cur_dir, f'acc-resid_post-test-{seed}.npy'))

    heads = np.load(os.path.join(cur_dir, f'acc-heads-test-{seed}.npy')).reshape(-1)
    mlps = np.load(os.path.join(cur_dir, f'acc-mlps-test-{seed}.npy'))
    accs_all = np.concatenate((mlps, heads), 0)

    results = {
        'acc_whole': whole_acc,
        'acc_mlps_heads': accs_all, 
    }

    if return_projs:
        heads = torch.load(os.path.join(cur_dir, f'projs-heads-test-{seed}.pt')).flatten(start_dim=0, end_dim=1)
        mlps = torch.load(os.path.join(cur_dir, f'projs-mlps-test-{seed}.pt'))
        # [n_comp, n_data, n_choices]
        logits_all = torch.cat((mlps, heads), 0)
        test_labels = np.load(os.path.join(cur_dir, 'test_label_ids.npy'))

        results['logits_mlps_heads'] = logits_all
        results['test_labels'] = test_labels
    return results


def get_ood_dir(tgt_task, tgt_dir):
    src_task = Demo_Dataset_Map[tgt_task]
    return tgt_dir.replace(tgt_task, src_task)
