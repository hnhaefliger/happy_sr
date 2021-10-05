import sys

from happy_sr import dataset
from happy_sr import model
from happy_sr import utils

n_mels = 64
hidden_dim = 256
dropout = 0.05
epochs = 10

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

    i += 1

print('loading dataset...')

train_loader = dataset.get_training_data(n_mels, batch_size=16, root='./cv-valid-train', tsv='train.tsv')
test_loader = dataset.get_training_data(n_mels, batch_size=16, root='./cv-valid-test', tsv='test.tsv')

print('done loading dataset.\n')
print('initializing model...')

sr_model = model.Model(n_mels, 29, hidden_dim, dropout)

print('model ready.\n')

if model.utils.device == 'cuda':
    model.cuda()

print('preparing optimizer...')

optimizer, scheduler = model.get_optimizer(sr_model, train_loader)
loss = model.get_loss()

print('optimizer ready.\n')
print('starting training...')

for epoch in range(1, epochs + 1):
    print(f'epoch {epoch}')
    utils.train(sr_model, optimizer, loss, train_loader)
    utils.test(sr_model, loss, test_loader)

    scheduler.step()

print('training complete.\n')
