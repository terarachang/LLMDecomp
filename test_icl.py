from utils import *






if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--dataset", type=str, default='sst2', help='test dataset')
    parser.add_argument("--mode", type=str, default='test')
    parser.add_argument("--format", type=int, default=1)
    parser.add_argument("--n_shots", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=40)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument('--seed_list', type=int, nargs='+')
    args = parser.parse_args()
    set_more_args(args, out_dir='ICL')

    model, tokenizer = load_model_tokenizer(args.model_name)
    for seed in tqdm(args.seed_list):
        test_sents, test_labels, test_label_ids, option_ids = prep_inputs(args, seed, tokenizer)

        all_pred_labels, all_probs = [], []
        for i in tqdm(range(0, len(test_sents), args.batch_size)):
            batch_sents = test_sents[i:i+args.batch_size]
            inputs = tokenizer(batch_sents, return_tensors="pt", padding=True)
            max_ctx_length = model.config.max_position_embeddings
            if inputs['input_ids'].shape[1] > max_ctx_length:
                print('inputs_len:', inputs['input_ids'].shape[1], 'max_len:', max_ctx_length)
                seq_lens = torch.LongTensor([max_ctx_length]*len(batch_sents))
                input_ids = inputs['input_ids'][:, -max_ctx_length:].to(device)
            else:
                seq_lens = inputs['attention_mask'].sum(1)
                input_ids = inputs['input_ids'].to(device)
            with torch.no_grad():
                out = model(input_ids).logits
                probs = out[range(len(out)), seq_lens-1].softmax(-1).cpu()

            if args.debug:
                top_toks = probs.argmax(-1)
                print('-'*100)
                print(repr(tokenizer.decode(top_toks)))
                print(tokenizer.batch_decode(torch.topk(probs[0], 4).indices))
                error = True
                for op in option_ids:
                    if op in top_toks: 
                        error = False
                        break
                if error:
                    print(top_toks)
                    print(option_ids)
                    pdb.set_trace()

            probs = probs[:,option_ids] # [bs, n_options]
            preds = probs.argmax(-1)
            all_pred_labels.append(preds)
            all_probs.append(probs)

        all_pred_labels = torch.cat(all_pred_labels).numpy()
        acc = (all_pred_labels == test_label_ids).mean()
        all_probs = torch.cat(all_probs)

        print(f'Acc: {acc:.1%}')
        option_ids = np.array(option_ids)
        all_pred_labels = tokenizer.batch_decode(option_ids[all_pred_labels])
        np.save(os.path.join(args.out_dir, f'acc-seed{seed}.npy'), acc)
        np.save(os.path.join(args.out_dir, f'all_pred_labels-seed{seed}.npy'), all_pred_labels)
        torch.save(all_probs, os.path.join(args.out_dir, f'all_probs_labels-seed{seed}.pt'))


