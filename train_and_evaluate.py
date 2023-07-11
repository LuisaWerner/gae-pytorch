import torch.nn.functional as F
import torch
import torch_geometric
import torch.backends.mps
from model import get_model
from preprocess import get_data
from time import time
from logger import *
from sklearn.metrics import classification_report


def train(model, loader, optimizer, device, criterion):
    model.train()
    total_loss = 0
    for i_batch, batch in enumerate(loader):
        batch.to(device)
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index)  # todo put corret activation ?
        loss = criterion(out, batch.edge_type) # todo
        total_loss += float(loss.item())

        loss.backward()
        optimizer.step()


@torch.no_grad()
def test(model, loader, criterion, device, evaluator):
    model.eval()
    # test loade rshould include the whole dataset
    for i, batch in enumerate(loader):
        batch .to(device)
        out = model(batch.x, batch.adj) # todo activation see above

    # todo
    # compute loss
    # train, valid, test loss
    # train, valid and test accuracy
    # return test_acc, valid_acc, train_acc, train_loss, valid_loss, test_loss


def run_experiment(args):
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
            data, train_data, val_data, test_data = get_data(args)
            # train_data.to(device), val_data.to(device), test_data.to(device)
            model = get_model(args, data).to(device)
            # model.reset_parameters()
            optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(args.adam_beta1, args.adam_beta2),
                                         eps=args.adam_eps, amsgrad=False, weight_decay=args.weight_decay)
            criterion = torch.nn.BCEWithLogitsLoss
            evaluator = classification_report # put metric todo
            run_logger = RunLogger(run, model, args)

            for epoch in range(args.epochs):
                # start = time()
                train(model, train_data, optimizer, device, criterion)
                # end = time()

                loss_and_metrics = test(model, criterion, device, evaluator) # todo output
                run_logger.update_per_epoch(**args) # todo

                # early stopping
                if run_logger.callback_early_stopping(epoch):
                    break

            loss_and_metrics_test = test(model, data, criterion, device, evaluator) # todo output
            run_logger.update_per_run(**args) # todo
            experiment_logger.add_run(run_logger)
            print(run_logger)

        experiment_logger.end_experiment()






