from utils import *
from config import Demo_Dataset_Map






if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--dataset", type=str, default='sst2', help='test dataset')
    parser.add_argument("--format", type=int, default=1)
    parser.add_argument("--n_shots", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument('--seed_list', type=int, nargs='+')
    args = parser.parse_args()
    set_more_args(args, out_dir='DECOMP')


    def rms_norm(x, scale):
        # x: [BS, d_model, n_heads], scale: [BS, 1, 1], model.model.norm.weight: [d_model]
        x = x.to(torch.float32)
        x = x * scale
        return model.model.norm.weight.unsqueeze(-1) * x.to(args.dtype)

    def unembed(h, ln_scale):
        h = rms_norm(h, ln_scale)
        #[n_options, d_model] x [BS, d_model, n_heads] -> [BS, n_options, n_heads]
        projs = torch.matmul(model.lm_head.weight[option_ids], h).cpu()
        return projs


    model, tokenizer = load_hooked_model_tokenizer(args.model_name)
    n_layers, n_heads, d_model = model.config.num_hidden_layers, model.config.num_attention_heads, model.config.hidden_size

    modes = ['test'] if args.dataset in Demo_Dataset_Map.keys() else ['dev', 'test']
    for mode in modes:
        args.mode = mode
        for seed in tqdm(args.seed_list):
            test_sents, test_labels, test_label_ids, option_ids = prep_inputs(args, seed, tokenizer)

            projs_mlp = torch.zeros((len(test_sents), n_layers, len(option_ids)), dtype=args.dtype)
            projs_head = torch.zeros((len(test_sents), n_layers, n_heads, len(option_ids)), dtype=args.dtype)
            projs_resid_post = torch.zeros((len(test_sents), len(option_ids)), dtype=args.dtype)
            for i in tqdm(range(0, len(test_sents), args.batch_size)):
                batch_sents = test_sents[i:i+args.batch_size]
                inputs = tokenizer(batch_sents, return_tensors="pt", padding=True)
                positions = inputs['attention_mask'].sum(1) - 1
                input_ids = inputs['input_ids'].to(device)
                bs = len(positions)

                out = model.run_with_cache(input_ids, last_positions = positions)
                ln_scale = out[1]['model.norm.hook_scale'][range(bs), positions].unsqueeze(-1)
                projs_resid_post[i:i+bs] = out[0].logits[range(bs), positions][:, option_ids].cpu()

                # Early Decode
                for l in range(n_layers):
                    mlp_name = f'model.layers.{l}.hook_mlp'
                    attn_name = f'model.layers.{l}.self_attn.hook_heads'
                    mlp_l = out[1][mlp_name].unsqueeze(-1)
                    attn_l = out[1][attn_name]
                    assert mlp_l.shape == (bs, d_model, 1), mlp_l.shape
                    assert attn_l.shape == (bs, d_model, n_heads), attn_l.shape
                    projs_mlp[i:i+bs, l] = unembed(mlp_l, ln_scale).squeeze()        # [BS, n_options]
                    projs_head[i:i+bs, l] = unembed(attn_l, ln_scale).transpose(1,2) # [BS, n_heads, n_options]
                del out # avoid oom

            projs_mlp = projs_mlp.permute((1, 0, 2))
            projs_head = projs_head.permute((1, 2, 0, 3))

            # get individual component accs
            calc_acc = lambda x: (x.argmax(-1).numpy() == test_label_ids).mean()

            accs_mlp = np.zeros((n_layers, ))
            accs_head = np.zeros((n_layers, n_heads))
            for l in range(n_layers):
                accs_mlp[l] = calc_acc(projs_mlp[l])
                for h_idx in range(n_heads):
                    accs_head[l, h_idx] = calc_acc(projs_head[l, h_idx])

            # save
            acc_full = calc_acc(projs_resid_post)
            print(f"FullModel: {acc_full:.1%}")
            np.save(os.path.join(args.out_dir, f'acc-resid_post-{args.mode}-{seed}.npy'), acc_full)
            torch.save(projs_resid_post, os.path.join(args.out_dir, f'projs-resid_post-{args.mode}-{seed}.pt'))

            print(f'TopHead: {accs_head.max():.1%}')
            np.save(os.path.join(args.out_dir, f'acc-heads-{args.mode}-{seed}.npy'), accs_head)
            torch.save(projs_head, os.path.join(args.out_dir, f'projs-heads-{args.mode}-{seed}.pt'))

            print(f'TopMLP: {accs_mlp.max():.1%}')
            np.save(os.path.join(args.out_dir, f'acc-mlps-{args.mode}-{seed}.npy'), accs_mlp)
            torch.save(projs_mlp, os.path.join(args.out_dir, f'projs-mlps-{args.mode}-{seed}.pt'))
            np.save(os.path.join(args.out_dir, f'{args.mode}_label_ids.npy'), test_label_ids)


