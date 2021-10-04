import torch


class BiRNN(torch.nn.Module):
    def __init__(self):
        super(BiRNN, self).__init__()
        self.rnn = torch.nn.RNN(128, 64, bidirectional=True)
        self.dropout = torch.nn.Dropout(0.05)

    def forward(self, x):
        x, _ = self.rnn(x)
        x = self.dropout(x)
        return x


class Model(torch.nn.Module):
    def __init__(self):
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

        self.rnn = BiRNN()
    
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
