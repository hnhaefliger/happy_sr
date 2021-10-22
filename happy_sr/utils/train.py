import torch
from tqdm import tqdm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(model, optimizer, loss_fn, dataset, metrics=[]):
    if device == 'cuda':
        model.cuda()

    model.train()
    progress_bar = tqdm(total=len(dataset))
    progress_bar.set_description(f'training')

    for batch_idx, (data, target, input_lengths, label_lengths) in enumerate(dataset):
        if device == 'cuda':
            data = data.cuda()
            target = target.cuda()

        output = model(data)
        output = torch.nn.functional.log_softmax(output, dim=2)

        output = output.transpose(0, 1)
        loss = loss_fn(output, target, input_lengths, label_lengths)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_idx % 10 == 0:
            info = {}
            for metric in metrics:
                info[metric.__name__] = metric(output, target)

        progress_bar.set_postfix(loss=f'{loss.item():.2f}', **info)
        progress_bar.update(1)

        del data, target, output, loss
        if device == 'cuda':
            torch.cuda.empty_cache()
