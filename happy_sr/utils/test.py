import torch
from tqdm import tqdm
import gc


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test(model, loss_fn, dataset, metrics=[]):
    model.to(device)
    model.eval()
    progress_bar = tqdm(total=len(dataset))
    progress_bar.set_description(f'evaluation')

    for batch_idx, (data, target, input_lengths, label_lengths) in enumerate(dataset):
        data = data.to(device)
        target = target.to(device)

        output = torch.nn.functional.log_softmax(model(data), dim=2).transpose(0, 1)

        loss = loss_fn(output, target, input_lengths, label_lengths)

        if batch_idx % 10 == 0:
            info = {}
            for metric in metrics:
                info[metric.__name__] = metric(output, target)

        progress_bar.set_postfix(loss=f'{loss.item():.2f}', **info)
        progress_bar.update(1)

        del data, target, output, loss
        gc.collect()
        torch.cuda.synchronize()
        if device == 'cuda':
            torch.cuda.empty_cache()
