import torch

class BiRNN(torch.nn.Module):
    def __init__(self, in_dim, out_dim, dropout):
        super(BiRNN, self).__init__()
        self.rnn = torch.nn.RNN(in_dim, out_dim, bidirectional=True)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x):
        x, _ = self.rnn(x)
        x = self.dropout(x)
        
        return x


class Model(torch.nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, dropout):
        super(Model, self).__init__()
        self.hidden1 = torch.nn.Sequential(
            torch.nn.Linear(in_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
        )

        self.rnn = BiRNN(hidden_dim, hidden_dim, dropout)
    
        self.hidden2 = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim*2, out_dim)
        )

        self.activate = torch.nn.functional.log_softmax

    def forward(self, x):
        sizes = x.size()
        x = torch.reshape(x, (sizes[0], sizes[1] * sizes[2], sizes[3]))
        x = torch.transpose(x, 1, 2)
        x = self.hidden1(x)
        x = self.rnn(x)
        x = self.hidden2(x)
        x = self.activate(x, dim=2)

        return x
