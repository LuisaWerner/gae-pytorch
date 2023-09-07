import importlib
import inspect
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as pyg
from torch_geometric.nn.conv import GCNConv, RGCNConv
from torch.nn import Parameter
from gae.layers import GraphConvolution


def get_model(args, data):
    """ instantiates the model specified in args """

    msg = f'{args.model} is not implemented. Choose a model in the list: ' \
          f'{[x[0] for x in inspect.getmembers(sys.modules["model"], lambda c: inspect.isclass(c) and c.__module__ == get_model.__module__)]}'
    module = importlib.import_module("model")
    try:
        _class = getattr(module, args.model)
    except AttributeError:
        raise NotImplementedError(msg)
    return _class(args, data)


class GCNModelVAE(nn.Module):
    def __init__(self, args, data):
        super(GCNModelVAE, self).__init__()
        self.hidden_dim = args.hidden_dim
        self.dropout = args.dropout
        self.gc1 = GCNConv(-1, self.hidden_dim)
        self.gc2 = GCNConv(self.hidden_dim, self.hidden_dim)
        self.gc3 = GCNConv(self.hidden_dim, self.hidden_dim)
        # self.gc1 = GraphConvolution(-1, self.hidden_dim, self.dropout, act=F.relu)
        # self.gc2 = GraphConvolution(self.hidden_dim, self.hidden_dim, self.dropout, act=self.act)
        # self.gc3 = GraphConvolution(self.hidden_dim, self.hidden_dim, self.dropout, act=self.act)
        self.dc = InnerProductDecoder(args)

    def encode(self, x, edge_index):
        z = F.dropout(x, self.dropout, self.training)
        z = self.gc1(z, edge_index)
        z = F.dropout(z, self.dropout, self.training)
        return self.gc2(z, edge_index), self.gc3(z, edge_index)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std) # random numbers of std's shape with mean 0 and variance 1
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x, edge_index):
        mu, logvar = self.encode(x, edge_index)
        z = self.reparameterize(mu, logvar)
        return self.dc(z), mu, logvar


class LinearClassifier(nn.Module):
    """ classifier that takes decoder output and makes multi label classification on edges """

    def __init__(self, args):
        super(LinearClassifier, self).__init__()
        self.hidden_dim = args.hidden_dim
        self.dropout = args.dropout
        self.num_classes = args.num_classes
        self.layers = torch.nn.ModuleList()
        self.layers.append(pyg.nn.Linear(-1, self.hidden_dim, bias=True))
        self.layers.append(pyg.nn.Linear(self.hidden_dim, self.hidden_dim, bias=True))
        self.layers.append(pyg.nn.Linear(self.hidden_dim, self.num_classes, bias=True))

    def forward(self, z):
        for _, layer in enumerate(self.layers[:-1]):
            z = F.relu(layer(z))
            z = F.dropout(z, self.dropout)
        z = self.layers[-1](z)
        return z


class InnerProductDecoder(nn.Module):
    """Decoder for using inner product for prediction."""

    def __init__(self, args):
        super(InnerProductDecoder, self).__init__()
        self.dropout = args.dropout
        self.num_classes = args.num_classes
        self.act = torch.sigmoid
        self.classifier = LinearClassifier(args)

    def forward(self, z):
        z = F.dropout(z, self.dropout, training=self.training) # why dropout in decoder?
        adj = self.act(torch.mm(z, z.t()))
        adj_3D = adj.unsqueeze(2).repeat(1, 1, self.num_classes)
        a_hat = self.classifier(adj_3D)
        return F.sigmoid(a_hat)

class RGCNEncoder(torch.nn.Module):
    def __init__(self, num_nodes, hidden_channels, num_relations):
        super().__init__()
        self.node_emb = Parameter(torch.empty(num_nodes, hidden_channels))
        self.conv1 = RGCNConv(hidden_channels, hidden_channels, num_relations,
                              num_blocks=5)
        self.conv2 = RGCNConv(hidden_channels, hidden_channels, num_relations,
                              num_blocks=5)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.node_emb)
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()

    def forward(self, edge_index, edge_type):
        x = self.node_emb
        x = self.conv1(x, edge_index, edge_type).relu_()
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv2(x, edge_index, edge_type)
        return x


class DistMultDecoder(torch.nn.Module):
    def __init__(self, num_relations, hidden_channels):
        super().__init__()
        self.rel_emb = Parameter(torch.empty(num_relations, hidden_channels))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.rel_emb)

    def forward(self, z, edge_index, edge_type):
        z_src, z_dst = z[edge_index[0]], z[edge_index[1]]
        rel = self.rel_emb[edge_type]
        return torch.sum(z_src * rel * z_dst, dim=1)
