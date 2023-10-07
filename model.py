import importlib
import inspect
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as pyg
import torch_geometric.nn.models.autoencoder
from torch_geometric.nn.conv import GCNConv, RGCNConv
from torch_geometric.nn import Linear
from torch.nn import Parameter, ModuleList
from gae.layers import GraphConvolution
from torch_geometric.nn.models.autoencoder import GAE
from abc import ABCMeta, abstractmethod, ABC


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


class Encoder(torch.nn.Module):

    @abstractmethod
    def __init__(self, args, data, **kwargs):
        super().__init__(**kwargs)
        if data.x is None:
            self.featureless = True
            self.node_embeddings = Parameter(torch.empty(data.num_nodes, args.hidden_dim))
        self.dropout = args.dropout
        self.num_hidden_layers = args.num_hidden_layers
        self.hidden_dim = args.hidden_dim

    @abstractmethod
    def reset_parameters(self):
        pass

    @abstractmethod
    def forward(self, **kwargs):
        pass


class Decoder(torch.nn.Module):

    @abstractmethod
    def __init__(self, *args, **kwargs):
        super().__init__(**kwargs)

    @abstractmethod
    def reset_parameters(self):
        pass

    @abstractmethod
    def forward(self, **kwargs):
        pass

class Classifier(nn.Module):
    """ classifier that takes decoder output and makes multi label classification on edges """

    def __init__(self, args, data):
        super(Classifier, self).__init__()
        self.hidden_dim = args.hidden_dim
        self.dropout = args.dropout
        self.num_classes = data.num_relations
        self.layers = torch.nn.ModuleList()
        self.layers.append(pyg.nn.Linear(-1, self.hidden_dim, bias=True))
        self.layers.append(pyg.nn.Linear(self.hidden_dim, self.hidden_dim, bias=True))
        self.layers.append(pyg.nn.Linear(self.hidden_dim, self.num_classes, bias=True))

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, z):
        for _, layer in enumerate(self.layers[:-1]):
            z = F.relu(layer(z))
            z = F.dropout(z, self.dropout)
        z = self.layers[-1](z)
        return z


class LinearEncoder(Encoder):
    """
    uses Linear layers as encoding.
    Makes only sense if there's node level feature information
    """

    def __init__(self, args, data, **kwargs):
        super().__init__(args, data, **kwargs)
        self.layers = ModuleList()
        self.layers.append(Linear(-1, args.hidden_dim))  # input layer
        for i in range(self.num_hidden_layers):
            self.layers.append(Linear(args.hidden_dim, args.hidden_dim))
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, batch):
        if self.featureless:
            z = self.layers[0](self.node_embeddings[batch.node_ids]).relu_()
        else:
            z = self.layers[0](batch.x).relu_()

        for layer in self.layers[1:]:
            z = F.dropout(z, p=self.dropout, training=self.training)
            z = layer(z).relu_()
        return z


class RelationalEncoder(Encoder):
    """ uses a GCN """

    def __init__(self, args, data, **kwargs):
        super().__init__(args, data, **kwargs)
        self.layers = ModuleList()
        self.layers.append(GCNConv(-1, args.hidden_dim))  # input layer
        for i in range(self.num_hidden_layers):
            self.layers.append(GCNConv(args.hidden_dim, args.hidden_dim))
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, batch):
        if self.featureless:
            z = self.layers[0](self.node_embeddings[batch.node_ids], batch.edge_index).relu_()
        else:
            z = self.layers[0](batch.x, batch.edge_index).relu_()

        for layer in self.layers[1:]:
            z = F.dropout(z, p=self.dropout, training=self.training)
            z = layer(z, batch.edge_index).relu_()
        return z


class MultiRelationalEncoder(Encoder):
    """ uses an R-GCN """

    def __init__(self, args, data, **kwargs):
        super().__init__(args, data, **kwargs)
        self.layers = ModuleList()
        self.layers.append(RGCNConv(-1,  args.hidden_dim, data.num_relations))  # input layer
        for i in range(self.num_hidden_layers):
            self.layers.append(RGCNConv(args.hidden_dim, args.hidden_dim, data.num_relations))
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, batch):
        if self.featureless:
            z = self.layers[0](self.node_embeddings[batch.node_ids], batch.edge_index, batch.edge_type).relu_()
        else:
            z = self.layers[0](batch.x, batch.edge_index, batch.edge_type).relu_()

        for layer in self.layers[1:]:
            z = F.dropout(z, p=self.dropout, training=self.training)
            z = layer(z, batch.edge_index, batch.edge_type).relu_()
        return z


class InnerProductDecoder(Decoder):
    # TODO VERIFY BEFORE USE, not tested in current setup and context
    """Decoder for using inner product for prediction."""

    def __init__(self, args, data, **kwargs):
        super().__init__(args, data)
        self.classifier = Classifier(args, data)
        self.dropout = args.dropout
        self.num_classes = data.num_relations
        self.act = torch.sigmoid # todo set to args

    def forward(self, z, batch):
        z = F.dropout(z, self.dropout, training=self.training)  # why dropout in decoder?
        adj = self.act(torch.mm(z, z.t()))
        adj_3D = adj.unsqueeze(2).repeat(1, 1, self.num_classes)
        a_hat = self.classifier(adj_3D)
        return a_hat


class HetDistMultDecoder(Decoder):
    """
    Decodes for multiple edge types
    We need a Parameter that decodes for each type separately
    Multi-label classficiation loss and activation :
    https://stackoverflow.com/questions/35400065/multilabel-text-classification-using-tensorflow/39472895#39472895
    """

    def __init__(self, args, data, **kwargs):
        super().__init__(args, data)
        self.rel_emb = Parameter(torch.empty(args.hidden_dim, data.num_relations + 1))  #
        self.reset_parameters()
        self.num_relations = data.num_relations

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.rel_emb)

    def forward(self, z, batch):
        z_src, z_dst = z[batch.pos_edge_index[0]], z[batch.pos_edge_index[1]]
        out = torch.matmul(z_src * z_dst, self.rel_emb)

        if hasattr(batch, 'neg_edge_index'):
            z_src_neg, z_dst_neg = z[batch.neg_edge_index[0]], z[batch.neg_edge_index[1]]
            neg_out = torch.matmul(z_src_neg * z_dst_neg, self.rel_emb)
            out = torch.cat([out, neg_out])

        return out
