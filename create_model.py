import sys

from happy_sr import dataset
from happy_sr import model
from happy_sr import utils

print('loading dataset...')

train_loader = dataset.get_training_data(batch_size=16, root='./cv-valid-train', tsv='train.tsv')
test_loader = dataset.get_training_data(batch_size=16, root='./cv-valid-test', tsv='test.tsv')

print('done loading dataset.\n')
print('initializing model...')

sr_model = model.Model(2, 2, 16, 29, 64)

print('model ready.\n')

if model.utils.device == 'cuda':
    model.cuda()

print('preparing optimizer...')

optimizer, scheduler = model.get_optimizer(sr_model, train_loader)
loss = model.get_loss()

print('optimizer ready.\n')
print('starting training...')

n_epoch = 2

for epoch in range(1, n_epoch + 1):
    print(f'epoch {epoch}')
    utils.train(sr_model, optimizer, loss, train_loader)
    utils.test(sr_model, loss, test_loader)

    scheduler.step()

print('training complete.\n')
