import sys
import torch

from happy_sr import dataset
from happy_sr import model
from happy_sr import utils

n_mels = 64
hidden_dim = 256
dropout = 0.05
epochs = 10
batch_size = 16
learning_rate = 1e-5
save = ''
load = ''

i = 0
while i < len(sys.argv):
    if sys.argv[i] == '--n-mels':
        n_mels = int(sys.argv[i+1])
        i += 1

    elif sys.argv[i] == '--hidden-dim':
        hidden_dim = int(sys.argv[i+1])
        i += 1

    elif sys.argv[i] == '--dropout':
        dropout = float(sys.argv[i+1])
        i += 1
    
    elif sys.argv[i] == '--epochs':
        epochs = int(sys.argv[i+1])
        i += 1

    elif sys.argv[i] == '--batch-size':
        batch_size = int(sys.argv[i+1])
        i += 1

    elif sys.argv[i] == '--lr':
        learning_rate = float(sys.argv[i+1])
        i += 1

    elif sys.argv[i] == '--save':
        save = sys.argv[i+1]
        i += 1

    elif sys.argv[i] == '--load':
        load = sys.argv[i+1]
        i += 1

    i += 1

print('loading dataset...')

train_loader = dataset.get_training_data(n_mels, batch_size=batch_size, root='./cv-valid-train', tsv='train.tsv')
test_loader = dataset.get_training_data(n_mels, batch_size=batch_size, root='./cv-valid-test', tsv='test.tsv')

print('done loading dataset.\n')
print('initializing model...')

sr_model = model.Model(n_mels, 29, hidden_dim, dropout)

if load:
    sr_model.load_state_dict(torch.load(load))

print('model ready.\n')

if model.utils.device == 'cuda':
    model.cuda()

print('preparing optimizer...')

optimizer, scheduler = model.get_optimizer(sr_model, train_loader, epochs, learning_rate)
loss = model.get_loss()

print('optimizer ready.\n')
print('starting training...')

for epoch in range(1, epochs + 1):
    print(f'epoch {epoch}')
    utils.train(sr_model, optimizer, loss, train_loader)#, metrics=[model.word_error_rate, model.char_error_rate])

    if save:
        torch.save(sr_model.state_dict(), save)

    utils.test(sr_model, loss, test_loader)

    scheduler.step()

print('training complete.\n')
