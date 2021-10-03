import torch

class CNNLayerNorm(torch.nn.Module):
    def __init__(self, n_feats):
        super(CNNLayerNorm, self).__init__()
        self.layer_norm = torch.nn.LayerNorm(n_feats)

    def forward(self, x):
        # x (batch, channel, feature, time)
        x = x.transpose(2, 3).contiguous()  # (batch, channel, time, feature)
        x = self.layer_norm(x)
        # (batch, channel, feature, time)
        return x.transpose(2, 3).contiguous()


class ResidualCNN(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride, dropout, n_feats):
        super(ResidualCNN, self).__init__()

        self.cnn1 = torch.nn.Conv2d(in_channels, out_channels,
                              kernel, stride, padding=kernel//2)
        self.cnn2 = torch.nn.Conv2d(out_channels, out_channels,
                              kernel, stride, padding=kernel//2)
        self.dropout1 = torch.nn.Dropout(dropout)
        self.dropout2 = torch.nn.Dropout(dropout)
        self.layer_norm1 = CNNLayerNorm(n_feats)
        self.layer_norm2 = CNNLayerNorm(n_feats)

    def forward(self, x):
        residual = x  # (batch, channel, feature, time)
        x = self.layer_norm1(x)
        x = torch.nn.functional.gelu(x)
        x = self.dropout1(x)
        x = self.cnn1(x)
        x = self.layer_norm2(x)
        x = torch.nn.functional.gelu(x)
        x = self.dropout2(x)
        x = self.cnn2(x)
        x += residual
        return x  # (batch, channel, feature, time)


class BidirectionalGRU(torch.nn.Module):
    def __init__(self, rnn_dim, hidden_size, dropout, batch_first):
        super(BidirectionalGRU, self).__init__()

        self.BiGRU = torch.nn.GRU(
            input_size=rnn_dim, hidden_size=hidden_size,
            num_layers=1, batch_first=batch_first, bidirectional=True)
        self.layer_norm = torch.nn.LayerNorm(rnn_dim)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x):
        x = self.layer_norm(x)
        x = torch.nn.functional.gelu(x)
        x, _ = self.BiGRU(x)
        x = self.dropout(x)
        return x
