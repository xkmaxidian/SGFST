import numpy as np
import torch
import torch.nn as nn
from model.layers import GraphConvolution, Attention
from model.encoder import GCNEncoder
from model.decoders import OmicsDecoder, CosineDecoder


class SGFST(nn.Module):
    def __init__(self, nfeat, nhid1, nhid2, dropout):
        super().__init__()
        self.GCN_S = GCNEncoder(nfeat, nhid1, nhid2, dropout)
        self.GCN_F = GCNEncoder(nfeat, nhid1, nhid2, dropout)
        self.dropout = dropout
        self.ATT = Attention(nhid2)
        self.MLP = nn.Sequential(
            nn.Linear(nhid2, nhid2)
        )
        self.omics_decoder = OmicsDecoder(nfeat, nhid1, nhid2)
        self.structure_decoder = CosineDecoder()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, adj_s, adj_f):
        h_s = self.GCN_S(x, adj_s)
        h_f = self.GCN_F(x, adj_f)
        h = torch.stack((h_s, h_f), dim=1)

        h, att = self.ATT(h)
        h = self.dropout(h)
        h = self.MLP(h)
        recon_adj = self.structure_decoder(h)
        shared_graph = torch.sigmoid(recon_adj)
        [pi, disp, mean] = self.omics_decoder(h, adj_s)
        # return h, recon_adj,pi, disp, mean, att
        return h, recon_adj,  pi, disp, mean, att,shared_graph


class NB(object):
    def __init__(self, theta=None, scale_factor=1.0):
        super(NB, self).__init__()
        self.eps = 1e-10
        self.scale_factor = scale_factor
        self.theta = theta

    def loss(self, y_true, y_pred, mean=True):
        y_pred = y_pred * self.scale_factor
        theta = torch.minimum(self.theta, torch.tensor(1e6))
        t1 = torch.lgamma(theta + self.eps) + torch.lgamma(y_true + 1.0) - torch.lgamma(y_true + theta + self.eps)
        t2 = (theta + y_true) * torch.log(1.0 + (y_pred / (theta + self.eps))) + (
                y_true * (torch.log(theta + self.eps) - torch.log(y_pred + self.eps)))
        final = t1 + t2
        final = _nan2inf(final)
        if mean:
            final = torch.mean(final)
        return final


def _nan2inf(x):
    return torch.where(torch.isnan(x), torch.zeros_like(x) + np.inf, x)


class ZINB(NB):
    def __init__(self, pi, ridge_lambda=0.0, **kwargs):
        super().__init__(**kwargs)
        self.pi = pi
        self.ridge_lambda = ridge_lambda

    def loss(self, y_true, y_pred, mean=True):
        scale_factor = self.scale_factor
        eps = self.eps
        theta = torch.minimum(self.theta, torch.tensor(1e6))
        nb_case = super().loss(y_true, y_pred, mean=False) - torch.log(1.0 - self.pi + eps)
        y_pred = y_pred * scale_factor
        zero_nb = torch.pow(theta / (theta + y_pred + eps), theta)
        zero_case = -torch.log(self.pi + ((1.0 - self.pi) * zero_nb) + eps)
        result = torch.where(torch.lt(y_true, 1e-8), zero_case, nb_case)
        ridge = self.ridge_lambda * torch.square(self.pi)
        result += ridge
        if mean:
            result = torch.mean(result)
        result = _nan2inf(result)
        return result
