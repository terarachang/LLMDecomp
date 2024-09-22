import pprint
from tqdm import tqdm
import torch.nn as nn
from utils_post import *


RES_dev = defaultdict(lambda: defaultdict(list))


class Calibrator(nn.Module):
    def __init__(self, n_choices):
        super(Calibrator, self).__init__()
        self.weights = nn.Parameter(torch.ones(n_choices))
        self.loss = nn.CrossEntropyLoss()

    def forward(self, logits, labels):
        #inputs: [bs, n_choices]
        p_ori = nn.Softmax(-1)(logits)
        new_logits = p_ori * self.weights
        loss = self.loss(new_logits, labels)
        return new_logits, loss


def build_model_optimizer(args, n_components):
    model = Calibrator(n_components)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    return model, optimizer


def load_data(hicl_dir, s, mode, n_labels=None):
    labels = np.load(os.path.join(hicl_dir, f'{mode}_label_ids.npy'))
    labels = torch.from_numpy(labels)
    logits = torch.load(os.path.join(hicl_dir, f'projs-resid_post-{mode}-{s}.pt')).float()
    if mode == 'dev':
        labels = labels[:n_labels] # [n_data, n_components, n_choices]
        logits = logits[:n_labels]
    print(mode, 'Logits:', logits.shape)
    return labels, logits


def test(model, logits, labels):
    with torch.no_grad():
        outputs, _ = model(logits, labels)
    preds = outputs.argmax(-1)
    acc = (preds == labels).numpy().mean()
    return acc

    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_labels", type=int, default=24)
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--out_dir", type=str, default="DECOMP")
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
                    labels_test, inputs_test = load_data(hicl_dir, s, 'test')
                    if args.oracle:
                        labels_dev, inputs_dev = labels_test, inputs_test
                    else:
                        labels_dev, inputs_dev = load_data(hicl_dir, s, 'dev', args.n_labels)

                    model, optimizer = build_model_optimizer(args, n_choices)

                    # start training
                    patience = 3
                    best_loss = 100.
                    for e in range(args.epochs):
                        optimizer.zero_grad()

                        _, loss = model(inputs_dev, labels_dev)

                        if e%100 == 0:
                            dev_acc = test(model, inputs_dev, labels_dev)
                            print(f'[{e:3d}] loss: {loss:.3f}, acc: {dev_acc:.1%}')
                            if dev_acc == 1.: break

                        loss.backward()
                        optimizer.step()

                        if best_loss <= loss:
                            patience -= 1
                        else:
                            best_loss = loss
                        if not patience: break

                    print("-"*100)
                    dev_acc = test(model, inputs_dev, labels_dev)
                    test_acc = test(model, inputs_test, labels_test)
                    print(task, f"Dev: {dev_acc:.1%}, Test: {test_acc:.1%}")
                    RES_dev[md][task].append(test_acc)
                    print("-"*100)

    fn = 'oracle-calib' if args.oracle else f'dev-{args.n_labels}-calib'
    summary = compact(RES_dev, fn)
    pprint.pp(summary)


