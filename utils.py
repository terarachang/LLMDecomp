import argparse
from collections import defaultdict
import json
import torch
import numpy as np
import pdb
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import itertools
from torch import Tensor
from tqdm import tqdm
from config import Demo_Dataset_Map
from transformers import AutoConfig


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
torch.set_grad_enabled(False)
HF_TOKEN = os.environ.get('HF_TOKEN')

def load_hooked_model_tokenizer(model_name, dtype='auto'):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        token=HF_TOKEN,
    ) 
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'

    def get_lm_class():
        if 'mistral' in model_name:
            from my_modeling_mistral import MistralForCausalLM as CausalLM
        elif 'llama' in model_name:
            from my_modeling_llama import LlamaForCausalLM as CausalLM
        else:
            raise NotImplementedError
        return CausalLM

    CausalLM = get_lm_class()
    model = CausalLM.from_pretrained(
        model_name,
        token=HF_TOKEN,
        torch_dtype=dtype,
    ) 
    model.to(device)
    return model, tokenizer

def load_model_tokenizer(model_name, dtype='auto'):
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        token=HF_TOKEN,
    ) 
    if not tokenizer.pad_token:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        token=HF_TOKEN,
        torch_dtype=dtype,
    ) 
    model.to(device)
    return model, tokenizer


def load_data(tokenizer, fn, option_ids=None, demo_str=''):
    print('-'*50)
    print(f"Preparing {fn}...")
    task = fn.split('_')[0]
    sents, labels, label_tokens = [], [], []
    path = os.path.join('data', task, fn)

    def debug_parse_input_output(txt): # for llama3
        '''
        There are bugs when setting `return_offsets_mapping=True` in llama3's tokenizer as of Sep 2024.
        When use llama3 as the LLM, use `debug_parse_input_output()` instead of `parse_input_output()`
        '''
        try:
            tokens = tokenizer.encode(txt, add_special_tokens=False)
            label_token = tokens[-1]
            x = tokenizer.decode(tokens[-3:-1])
            label = txt.rsplit(x, 1)[1]
            input_str = txt.rsplit(label, 1)[0]
        except:
            if '...' in x: x = x.replace('...', ' ...')
            label = txt.rsplit(x, 1)[1]
            input_str = txt.rsplit(label, 1)[0]
        return input_str, label, label_token

    def parse_input_output(txt):
        # Note: we chose label words that only have single token (tokens[-1])
        x = tokenizer(txt, add_special_tokens=False, return_offsets_mapping=True)
        label_token = x['input_ids'][-1]
        lb_start, lb_end = x['offset_mapping'][-1]
        label = txt[lb_start: lb_end]
        input_str = txt[:lb_start]
        return input_str, label, label_token

    with open(path, 'r') as f:
        for line in f:
            dp = json.loads(line) # dict
            # test & have instructions
            if 'definition' in dp:
                if demo_str:
                    demo_str = dp['definition'] + '\n\n' + demo_str
                continue

            input_str, label, label_token = parse_input_output(dp['text'])
            sents.append(demo_str+input_str)
            labels.append(label)
            label_tokens.append(label_token)

        if not option_ids: option_ids = list(set(label_tokens))
        assert len(option_ids) == len(dp['options']), "bugs in label tokens"
        assert len(set(labels)) == len(dp['options']), "bugs in labels (txt)"

        labels = np.array(labels)
        label_ids = np.array([option_ids.index(t) for t in label_tokens])

    return sents, labels, label_ids, option_ids # [N, seq_len], [N], [N], [n_classes]


def set_more_args(args, out_dir):
    if 'nli' in args.dataset and args.n_shots == 4: args.n_shots = 3
    if args.n_shots >= 24: args.batch_size = 1
    args.out_dir = os.path.join(out_dir, args.model_name, args.dataset, f'{args.n_shots}shots', f'f{args.format}')
    os.makedirs(args.out_dir, exist_ok=True)

    config = AutoConfig.from_pretrained(args.model_name, token=HF_TOKEN)
    args.dtype = config.torch_dtype

    print(args)
    print('-'*50)


def get_model_attr(model_name):
    config = AutoConfig.from_pretrained(model_name)
    n_heads, n_layers, d_model = config.num_attention_heads, config.num_hidden_layers, config.hidden_size
    d_head = d_model // n_heads
    return n_heads, n_layers, d_head, d_model


def sample_demo(sents, labels, seed, n_shots, n_classes, merge=True, label_ids=None):
    np.random.seed(seed) 
    sents = sents[:n_shots] # data in the dev file are ordered in a balanced way
    labels = labels[:n_shots]
    assert len(set(labels)) == n_classes
    if merge:
        while True: # random order but avoid recency bias
            indices = np.arange(n_shots)
            np.random.shuffle(indices)
            if labels[indices[-1]] != labels[indices[-2]]:
                break

        demo_str = ''
        for i in indices:
            demo_str += f'{sents[i]}{labels[i]}\n'

        return demo_str
    else:
        return sents, label_ids[:n_shots]


def print_data(data):
    print('-'*50)
    for dp in data:
        print(dp)
    print('-'*50)


def prep_inputs(args, seed, tokenizer):
    # build demonstrations
    demo_dataset = Demo_Dataset_Map[args.dataset] if args.dataset in Demo_Dataset_Map else args.dataset
    dev_sents, dev_labels, _, option_ids = load_data(tokenizer, f'{demo_dataset}_dev-f{args.format}-s{seed}.jsonl')
    print('decoded option_ids:', tokenizer.batch_decode(option_ids))

    demo_str = sample_demo(dev_sents, dev_labels, seed, args.n_shots, len(option_ids))

    # add demonstrations to the test examples # input option_ids to fix the order of options
    if args.mode == 'dev':
        test_sents, test_labels, test_label_ids, _ = \
            load_data(tokenizer, f'{args.dataset}_dev-f{args.format}-s{seed}.jsonl', option_ids, demo_str)
        test_sents, test_labels, test_label_ids = test_sents[args.n_shots:], test_labels[args.n_shots:], test_label_ids[args.n_shots:]
    else:
        test_sents, test_labels, test_label_ids, _ = \
            load_data(tokenizer, f'{args.dataset}_test-f{args.format}.jsonl', option_ids, demo_str)
    print(f'# {args.mode} data:', len(test_sents))
    print_data([tokenizer.decode(tokenizer.encode(test_sents[0]))])

    return test_sents, test_labels, test_label_ids, option_ids
