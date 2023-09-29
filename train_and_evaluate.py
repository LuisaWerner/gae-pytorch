import torch.nn.functional as F
import torch
from tqdm import tqdm
import torch_geometric
import torch.backends.mps
from model import get_model, RGCNEncoder, DistMultDecoder, HetDistMultDecoder, MLPEncoder
from preprocess import get_data
from torch_geometric.nn import GAE
from torch_geometric.utils import negative_sampling
from utils import compute_mrr
from logger import *
from preprocess import SubgraphSampler
from sklearn.metrics import roc_auc_score


def train(model, data, optimizer, device, args):
    model.train()
    optimizer.zero_grad()

    # do negative sampling here and then sample per batch
    train_loader = SubgraphSampler(data.train_data, shuffle=True, neg_sampling_per_type=False, neg_sampling_ratio=args.neg_sampling_ratio)
    regularize = False  # todo

    total_loss = 0
    for i_batch, batch in enumerate(tqdm(train_loader)):
        batch.to(device)
        z = model.encode(batch)
        out = model.decode(z, batch)  # pos and neg edges
        loss = F.binary_cross_entropy_with_logits(out, batch.edge_label)

        if args.regularize:
            reg_loss = z.pow(2).mean() + model.decoder.rel_emb.pow(2).mean()  # regularization # todo do we need this?
            loss = loss + 1e-2 * reg_loss

        total_loss += loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
        optimizer.step()

    # todo we do not even need a return statement?
    return float(total_loss)


@torch.no_grad()
def test(model, data, device, args):
    loader = SubgraphSampler(data, shuffle=False, neg_sampling_per_type=False, neg_sampling_ratio=args.neg_sampling_ratio)
    model.eval()

    outs = []
    ground_truths = []
    for i_batch, batch in enumerate(tqdm(loader)):
        batch.to(device)
        z = model.encode(batch)
        outs.append(model.decode(z, batch).sigmoid())  # todo is sigmoid correct  ?
        ground_truths.append(batch.edge_label)

    all_outs = torch.cat(outs, dim=0)
    all_ground_truth = torch.cat(ground_truths, dim=0)
    auc = roc_auc_score(all_ground_truth.cpu().numpy(), all_outs.cpu().numpy())

    # todo which metric do we use?
    # valid_mrr = compute_mrr(z, data.val_edge_index, data.val_edge_type, data, model)
    # test_mrr = compute_mrr(z, data.test_edge_index, data.test_edge_type, data, model)
    # return valid_mrr, test_mrr
    return auc


def run_conf(args):
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

    # experiment_logger = ExperimentLogger(args)
    for run in range(args.runs):
        data = get_data(args).to(device)
        model = GAE(
            encoder=MLPEncoder(args),
            decoder=HetDistMultDecoder(num_relations=data.num_relations, hidden_channels=args.hidden_dim),
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate,
                                     betas=(args.adam_beta1, args.adam_beta2),
                                     eps=args.adam_eps, amsgrad=False, weight_decay=args.weight_decay)

        # run_logger = RunLogger(run, model, args)
        for epoch in range(args.epochs):
            print(f'Run {run}, Epoch {epoch}')
            loss = train(model, data, optimizer, device, args)
            print(f'Loss: {loss:.4f}')
            train_auc = test(model, data.train_data, device, args)
            val_auc = test(model, data.val_data, device, args)
            test_auc = test(model, data.test_data, device, args)
            print(f'Train AUC: {train_auc}, Val AUC: {val_auc}, Test AUC: {test_auc}')

            # todo which metric to use
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
