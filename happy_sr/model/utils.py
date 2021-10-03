import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_optimizer(model, loader):
    optimizer = torch.optim.AdamW(model.parameters(), 1e-6)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=1e-6, steps_per_epoch=int(len(loader)), epochs=10, anneal_strategy='linear')

    return optimizer, scheduler


def get_loss():
    loss = torch.nn.CTCLoss(blank=28).to(device)

    return loss
