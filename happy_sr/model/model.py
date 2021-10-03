import torch
from .layers import *


class Model(torch.nn.Module):
    def __init__(self, n_cnn_layers, n_rnn_layers, rnn_dim, n_class, n_feats, stride=2, dropout=0.1):
        super(Model, self).__init__()
        n_feats = n_feats//2
        # cnn for extracting heirachal features
        self.cnn = torch.nn.Conv2d(1, 32, 3, stride=stride, padding=3//2)

        # n residual cnn layers with filter size of 32
        self.rescnn_layers = torch.nn.Sequential(*[
            ResidualCNN(32, 32, kernel=3, stride=1,
                        dropout=dropout, n_feats=n_feats)
            for _ in range(n_cnn_layers)
        ])
        self.fully_connected = torch.nn.Linear(n_feats*32, rnn_dim)
        self.birnn_layers = torch.nn.Sequential(*[
            BidirectionalGRU(rnn_dim=rnn_dim if i == 0 else rnn_dim*2,
                             hidden_size=rnn_dim, dropout=dropout, batch_first=i == 0)
            for i in range(n_rnn_layers)
        ])
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(rnn_dim*2, rnn_dim),  # birnn returns rnn_dim*2
            torch.nn.GELU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(rnn_dim, n_class)
        )

    def forward(self, x):
        x = self.cnn(x)
        x = self.rescnn_layers(x)
        sizes = x.size()
        x = x.view(sizes[0], sizes[1] * sizes[2],
                   sizes[3])  # (batch, feature, time)
        x = x.transpose(1, 2)  # (batch, time, feature)
        x = self.fully_connected(x)
        x = self.birnn_layers(x)
        x = self.classifier(x)
        return x
