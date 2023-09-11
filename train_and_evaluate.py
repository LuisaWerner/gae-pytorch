
import torch.nn.functional as F
import torch
import torch_geometric
import torch.backends.mps
from model import get_model, RGCNEncoder, DistMultDecoder
from preprocess import get_data
from torch_geometric.nn import GAE
from torch_geometric.utils import negative_sampling
from utils import compute_mrr
from logger import *


def train(model, data, optimizer):
    model.train()
    optimizer.zero_grad()

    z = model.encode(data.x, data.edge_index, data.edge_type)

    pos_out = model.decode(z, data.train_edge_index, data.train_edge_type)

    neg_edge_index = negative_sampling(data.edge_index, data.num_nodes, num_neg_samples=len(data.train_edge_index[1])) # could be done in sampling ?
    neg_out = model.decode(z, neg_edge_index, data.train_edge_type)

    out = torch.cat([pos_out, neg_out])
    gt = torch.cat([torch.ones_like(pos_out), torch.zeros_like(neg_out)])
    cross_entropy_loss = F.binary_cross_entropy_with_logits(out, gt)
    reg_loss = z.pow(2).mean() + model.decoder.rel_emb.pow(2).mean() # regularization
    loss = cross_entropy_loss + 1e-2 * reg_loss

    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
    optimizer.step()

    return float(loss)

@torch.no_grad()
def test(model, data):
    model.eval()
    z = model.encode(data.x, data.edge_index, data.edge_type)

    # todo how does mrr actually work ?
    valid_mrr = compute_mrr(z, data.val_edge_index, data.val_edge_type, data, model)
    test_mrr = compute_mrr(z, data.test_edge_index, data.test_edge_type, data, model)

    return valid_mrr, test_mrr

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
        for run in range(args.runs):
            data = get_data(args).to(device)
            model = GAE(
                RGCNEncoder(data.num_nodes, 500, num_relations=len(data.edge_type_dict.keys())),
                DistMultDecoder(num_relations=30, hidden_channels=500),
            ).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(args.adam_beta1, args.adam_beta2),
                                         eps=args.adam_eps, amsgrad=False, weight_decay=args.weight_decay)

            for epoch in range(args.epochs):
                loss = train(model, data, optimizer)
                print(f'Epoch: {epoch:05d}, Loss: {loss:.4f}')
                valid_mrr, test_mrr = test(model, data)
                print(f'Val MRR: {valid_mrr:.4f}, Test MRR: {test_mrr:.4f}')






