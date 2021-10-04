import torch


class Model(torch.nn.Module):
    def __init__(self, n_cnn_layers, n_rnn_layers, rnn_dim, n_class, n_feats, stride=2, dropout=0.1):
        super(Model, self).__init__()
        self.hidden1 = torch.nn.Sequential(
            torch.nn.Linear(1024, 1024),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.05),
            torch.nn.Linear(1024, 1024),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.05),
            torch.nn.Linear(1024, 1024),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.05),
        )
        self.rnn = torch.nn.Sequential(
            torch.nn.RNN(512, 512, bidirectional=True),
            torch.nn.Dropout(0.50),
        )
        self.hidden2 = torch.nn.Sequential(
            torch.nn.Linear(512, 29)
        )

    def forward(self, x):
        x = x.transpose(1, 2)  # (batch, time, feature)
        x = self.hidden1(x)
        x = x.transpose(1, 2)  # (batch, feature, time)
        x = self.rnn(x)
        x = x.transpose(1, 2)  # (batch, time, feature)
        x = self.hidden2(x)
        x = x.transpose(1, 2)  # (batch, feature, time)
        return x
