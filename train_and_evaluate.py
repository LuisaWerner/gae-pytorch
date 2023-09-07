import sklearn.metrics
import torch.nn.functional as F
import torch
import torch_geometric
import torch.backends.mps
from model import get_model, RGCNEncoder, DistMultDecoder
from preprocess import get_data
from torch_geometric.nn import GAE
from time import time
from logger import *
from sklearn.metrics import classification_report


def train(model, loader, optimizer, device, criterion):
    model.train()
    total_loss = 0
    for i_batch, batch in enumerate(loader):
        print(f'Batch {i_batch} of {len(loader)}')
        batch.to(device)
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index)[0]
        loss = criterion(out, batch.edge_labels.float(), reduction='mean')
        total_loss += float(loss.item())

        loss.backward()
        optimizer.step()


@torch.no_grad()
def test(model, loader, criterion, device, evaluator):
    model.eval()
    # test loader should include the whole dataset
    outs, labels = [], []
    for i, batch in enumerate(loader):
        batch.to(device)
        out = model(batch.x, batch.edge_index)[0]  # todo put corret activation ?
        # loss = criterion(out, batch.edge_labels.float(), reduction='mean')
        outs.append(out.cpu()) # todo: outs have different shapes because the number of nodes is different
        labels.append(batch.edge_labels)
    all_outs = torch.cat(outs, dim=0)
    labels = torch.cat(labels, dim=0)
    # train_score = evaluator(all_outs[train_mask], labels[train_mask]) ... todo





# todo
    # compute loss
    # train, valid, test loss
    # train, valid and test accuracy
    # return test_acc, valid_acc, train_acc, train_loss, valid_loss, test_loss


def run_experiment(args):
    """
    helpful article on multi-label classification
    https://www.kdnuggets.com/2023/03/multilabel-nlp-analysis-class-imbalance-loss-function-approaches.html#:~:text=In%20the%20context%20of%20using,loss%20as%20the%20loss%20function.
    """
    torch_geometric.seed_everything(args.seed)
    if args.mps:
        print(f"MPS backend available {torch.backends.mps.is_available()}")
        device = torch.device("mps")
    else:
        device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
        device = torch.device(device)
        print(f'Cuda available? {torch.cuda.is_available()}, Number of devices: {torch.cuda.device_count()}')

        print('Start training')
        experiment_logger = ExperimentLogger(args)

        # evaluator = Evaluator(args)

        for run in range(args.runs):
            data = get_data(args)
            model = GAE(
                RGCNEncoder(data.num_nodes, 500, num_relations=len(data.edge_type_dict.keys())),
                DistMultDecoder(num_relations=len(data.edge_type_dict.keys())// 2, hidden_channels=500),
            ).to(device)
            # model = get_model(args, data).to(device)
            # model.reset_parameters()
            optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(args.adam_beta1, args.adam_beta2),
                                         eps=args.adam_eps, amsgrad=False, weight_decay=args.weight_decay)
            criterion = F.binary_cross_entropy  # or with logits ?
            evaluator = sklearn.metrics.f1_score # classification_report["f1score"] # put metric todo
            run_logger = RunLogger(run, model, args)

            for epoch in range(args.epochs):
                train(model, train_data, optimizer, device, criterion)

                loss_and_metrics = test(model, train_data, criterion, device, evaluator) # todo just put here for debug
                run_logger.update_per_epoch(**args) # todo

                # early stopping
                if run_logger.callback_early_stopping(epoch):
                    break

            loss_and_metrics_test = test(model, data, criterion, device, evaluator) # todo output
            run_logger.update_per_run(**args) # todo
            experiment_logger.add_run(run_logger)
            print(run_logger)

        experiment_logger.end_experiment()






