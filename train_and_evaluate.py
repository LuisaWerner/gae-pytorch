import torch.nn.functional as F
import torch
from tqdm import tqdm
import torch_geometric
import torch.backends.mps
from model import *
from preprocess import get_data
from torch_geometric.nn import GAE
from torch_geometric.utils import negative_sampling
from utils import *
from logger import *
from preprocess import SubgraphSampler
from sklearn.metrics import roc_auc_score
from torch_geometric.data import Data


def train(model: torch.nn.Module, data: Data, optimizer: torch.optim.Optimizer, device, args) -> float:
    """
    Conducts one epoch in training
    @param model: prediction model
    @param data: training data object
    @param optimizer
    @param device: cpu or gpu (number)
    @param args: conf arguments
    @returns: epoch loss
    """

    model.train()
    optimizer.zero_grad()

    train_loader = SubgraphSampler(data.train,
                                   batch_size=args.batch_size,
                                   shuffle=True,
                                   neg_sampling_per_type=False,
                                   neg_sampling_ratio=args.neg_sampling_ratio)

    total_loss = 0
    # for i_batch, batch in enumerate(tqdm(train_loader)):
    for i_batch, batch in enumerate(train_loader):
        batch.to(device)
        z = model.encode(batch)
        out = F.sigmoid(model.decode(z, batch))
        # todo we could also use BCE_with_logits and remove activation function in output layer
        loss = F.binary_cross_entropy(out, batch.edge_label)

        if args.regularize:
            reg_loss = z.pow(2).mean() + model.decoder.rel_emb.pow(2).mean()
            loss = loss + 1e-2 * reg_loss

        total_loss += loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()

    return float(total_loss)


@torch.no_grad()
def test(model: torch.nn.Module, data: Data, device, args):
    """
    Conducts one epoch in training
    @param model: prediction model
    @param data: training data object
    @param device: cpu or gpu (number)
    @param args: conf arguments
    @returns: evaluation metric, here: AUC
    """
    loader = SubgraphSampler(data,
                             batch_size=args.batch_size,
                             shuffle=False,
                             neg_sampling_per_type=False,
                             neg_sampling_ratio=args.neg_sampling_ratio)
    model.eval()

    outs = []
    ground_truths = []
    for i_batch, batch in enumerate(loader):
    # for i_batch, batch in enumerate(tqdm(loader)):
        batch.to(device)
        z = model.encode(batch)
        outs.append(model.decode(z, batch).sigmoid())
        ground_truths.append(batch.edge_label)

    all_outs = torch.cat(outs, dim=0)
    all_ground_truth = torch.cat(ground_truths, dim=0)

    auc = roc_auc_score(all_ground_truth.cpu().numpy(), all_outs.cpu().numpy())
    metrics = Evaluator(all_ground_truth, all_outs, model, data).compute_rank()

    # todo alternatively compute mrr/h@k?
    return auc


def run_conf(args):
    """
    runs experiments defined in a conf file
    (multiple runs)
    """
    torch_geometric.seed_everything(args.seed)
    if args.mps:
        print(f"MPS backend available {torch.backends.mps.is_available()}")
        device = torch.device("mps")
    else:
        device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
        device = torch.device(device)
        print(f'Cuda available? {torch.cuda.is_available()}, Number of devices: {torch.cuda.device_count()}')

    # experiment_logger = ExperimentLogger(args)
    for run in range(args.runs):
        data = get_data(args).to(device)
        model = GAE(
            encoder=RelationalEncoder(args, data),
            decoder=HetDistMultDecoder(args, data),
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate,
                                     betas=(args.adam_beta1, args.adam_beta2),
                                     eps=args.adam_eps, amsgrad=False, weight_decay=args.weight_decay)

        # run_logger = RunLogger(run, model, args)
        for epoch in range(args.epochs):
            print(f'Run {run}, Epoch {epoch}')
            loss = train(model, data, optimizer, device, args)
            print(f'Loss: {loss:.4f}')
            train_auc = test(model, data.train, device, args)
            val_auc = test(model, data.valid, device, args)
            test_auc = test(model, data.test, device, args)
            print(f'Train AUC: {train_auc}, Val AUC: {val_auc}, Test AUC: {test_auc}')

            # todo mrr/H@k metrics?
            # valid_mrr, test_mrr = test(model, data)
            # print(f'Val MRR: {valid_mrr:.4f}, Test MRR: {test_mrr:.4f}')
            # run_logger.update_per_epoch(**args) # todo

            # early stopping
            # if run_logger.callback_early_stopping(epoch):
            #     break

        # run_logger.update_per_run(**args) # todo
        # experiment_logger.add_run(run_logger)
        # print(run_logger)

    # experiment_logger.end_experiment()
