import torch


class Model(torch.nn.Module):
    def __init__(self, n_cnn_layers, n_rnn_layers, rnn_dim, n_class, n_feats, stride=2, dropout=0.1):
        super(Model, self).__init__()
        self.hidden1 = torch.nn.Sequential(
            torch.nn.Linear(64, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.05),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.05),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.05),
        )
        self.rnn = torch.nn.Sequential(
            torch.nn.RNN(128, 128, bidirectional=True),
            torch.nn.Dropout(0.50),
        )
        self.hidden2 = torch.nn.Sequential(
            torch.nn.Linear(128, 29)
        )

    def forward(self, x):
        sizes = x.size()
        x = torch.reshape(x, (sizes[0], sizes[1] * sizes[2], sizes[3]))
        x = torch.transpose(x, 1, 2)
        x = self.hidden1(x)
        x = self.rnn(x)
        x = self.hidden2(x)
        return x
