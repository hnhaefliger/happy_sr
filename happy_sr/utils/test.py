import torch
from tqdm import tqdm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test(model, loss_fn, dataset):
    model.eval()
    progress_bar = tqdm(total=len(dataset))
    progress_bar.set_description(f'evaluation')

    for batch_idx, (data, target, input_lengths, label_lengths) in enumerate(dataset):
        data = data.to(device)
        target = target.to(device)

        output = model(data)
        output = torch.nn.functional.log_softmax(output, dim=2)
        output = output.transpose(0, 1)

        loss = loss_fn(output, target, input_lengths, label_lengths)

        progress_bar.set_postfix(loss=f'{loss.item():.2f}')

        progress_bar.update(1)
