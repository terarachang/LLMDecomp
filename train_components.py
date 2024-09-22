import pprint
from tqdm import tqdm
import torch.nn as nn
from utils_post import *


RES_dev = defaultdict(lambda: defaultdict(list))


class Classifier(nn.Module):
    def __init__(self, n_components):
        super(Classifier, self).__init__()
        self.weights = nn.Parameter(torch.ones(n_components,1))
        self.loss = nn.CrossEntropyLoss()

    def forward(self, inputs, labels):
        #inputs: [bs, n_components, n_choices]
        out = inputs * torch.clip(self.weights, 0, 1)
        out = out.sum(1)
        clf_loss = self.loss(out, labels)
        return out, clf_loss


def build_model_optimizer(args, n_components):
    model = Classifier(n_components)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    return model, optimizer


def load_data(hicl_dir, s, mode, n_labels=None):
    proj_mlp_fn = f'{hicl_dir}/projs-mlps-{mode}-{s}.pt'
    proj_head_fn = f'{hicl_dir}/projs-heads-{mode}-{s}.pt'
    projs_head = torch.load(proj_head_fn)
    projs_head = projs_head.view(-1, projs_head.shape[-2], projs_head.shape[-1])
    projs_mlp = torch.load(proj_mlp_fn)
    projs = torch.cat((projs_head, projs_mlp), 0) # [n_components, n_data, n_choices]
    projs = projs.permute(1,0,2).float()          # [n_data, n_components, n_choices]
    labels = np.load(os.path.join(hicl_dir, f'{mode}_label_ids.npy'))
    labels = torch.from_numpy(labels)
    if mode == 'dev':
        projs = projs[:n_labels] # [n_data, n_components, n_choices]
        labels = labels[:n_labels]
    assert len(labels) == len(projs)
    print(mode, 'Projs:', projs.shape)
    return labels, projs


def test(model, projs, labels):
    with torch.no_grad():
        outputs, _ = model(projs, labels)
    preds = outputs.argmax(-1)
    acc = (preds == labels).numpy().mean()
    return acc

    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_labels", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--out_dir", type=str, default="DECOMP")
    parser.add_argument("--lambda_l1", type=float, default=1e-2)
    parser.add_argument("--lr", type=float, default=5e-2)
    parser.add_argument("--oracle", action="store_true")
    args = parser.parse_args()
    print(args)


    for (model, size) in [('llama', 7), ('llama', 13), ('mistral', 7)]:
        md = f'{model}-{size}B'
        model_dir = model_map(model, size)

        for task in Tasks.keys():
            n_shots = 3 if task == 'mnli' else 4
            n_choices = Tasks[task] 

            for form_i in [1,2,3]:
                for s in range(5):
                    # load data
                    hicl_dir = f'{args.out_dir}/{model_dir}/{task}/{n_shots}shots/f{form_i}'
                    labels_test, projs_test = load_data(hicl_dir, s, 'test')
                    if args.oracle:
                        labels_dev, projs_dev = labels_test, projs_test
                    else:
                        labels_dev, projs_dev = load_data(hicl_dir, s, 'dev', args.n_labels)

                    # build model
                    n_components = projs_dev.shape[1]
                    model, optimizer = build_model_optimizer(args, n_components)

                    # start training
                    patience = 3
                    best_loss = 100.
                    for e in range(args.epochs):
                        optimizer.zero_grad()

                        _, clf_loss = model(projs_dev, labels_dev)
                        l1_loss = args.lambda_l1 * torch.norm(model.weights, 1)
                        loss = clf_loss + l1_loss
                        loss.backward()
                        optimizer.step()

                        if e%100 == 0:
                            dev_acc = test(model, projs_dev, labels_dev)
                            print(f'[{e:3d}] Clf: {clf_loss:.3f}, L1: {l1_loss:.3f}, dev: {dev_acc:.1%}')
                            if dev_acc == 1.: break

                        if best_loss <= clf_loss:
                            patience -= 1
                        else:
                            best_loss = clf_loss
                        if not patience: break

                    print("-"*100)
                    dev_acc = test(model, projs_dev, labels_dev)
                    test_acc = test(model, projs_test, labels_test)
                    print(task, f"Dev: {dev_acc:.1%}, Test: {test_acc:.1%}")
                    RES_dev[md][task].append(test_acc)
                    print("-"*100)

    fn = 'oracle-components' if args.oracle else f'dev-{args.n_labels}-components'
    summary = compact(RES_dev, fn)
    pprint.pp(summary)


