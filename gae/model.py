import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric as pyg
from gae.layers import GraphConvolution


class GCNModelVAE(nn.Module):
    def __init__(self, input_feat_dim, hidden_dim1, hidden_dim2, dropout):
        super(GCNModelVAE, self).__init__()
        self.gc1 = GraphConvolution(input_feat_dim, hidden_dim1, dropout, act=F.relu)
        self.gc2 = GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=lambda x: x)
        self.gc3 = GraphConvolution(hidden_dim1, hidden_dim2, dropout, act=lambda x: x)
        self.dc = InnerProductDecoder(dropout, act=lambda x: x)

    def encode(self, x, adj):
        hidden1 = self.gc1(x, adj)
        return self.gc2(hidden1, adj), self.gc3(hidden1, adj)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x, adj):
        mu, logvar = self.encode(x, adj)
        z = self.reparameterize(mu, logvar)
        return self.dc(z), mu, logvar


class LinearClassifier(nn.Module):
    """ classifier that takes decoder output and makes multi label classification on edges """

    def __init__(self, hidden_dim, dropout, num_classes):
        super(LinearClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.layers = torch.nn.ModuleList()
        self.layers.append(pyg.nn.Linear(-1, self.hidden_dim, bias=True))
        self.layers.append(pyg.nn.Linear(self.hidden_dim, self.hidden_dim, bias=True))
        self.layers.append(pyg.nn.Linear(self.hidden_dim, num_classes, bias=True))

    def forward(self, z):
        for _, layer in enumerate(self.layers[:-1]):
            z = F.relu(layer(z))
            z = F.dropout(z, self.dropout)
        z = self.layers[-1](z)
        return z


class InnerProductDecoder(nn.Module):
    """Decoder for using inner product for prediction."""

    def __init__(self, dropout, num_classes=10, act=torch.sigmoid):
        super(InnerProductDecoder, self).__init__()
        self.dropout = dropout
        self.act = act
        self.num_classes = num_classes
        self.classifier = LinearClassifier(hidden_dim=128, dropout=0.5, num_classes=self.num_classes)

    def forward(self, z):
        z = F.dropout(z, self.dropout, training=self.training)
        adj = self.act(torch.mm(z, z.t()))
        adj_3D = adj.unsqueeze(2).repeat(1, 1, self.num_classes)
        a_hat = self.classifier(adj_3D)
        return F.sigmoid(a_hat)
