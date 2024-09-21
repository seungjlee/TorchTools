import torch
import tqdm.auto as tqdm
tqdm.tqdm.format_num = staticmethod(lambda n: f"{n:.5f}" if isinstance(n, float) and abs(n) > 1e-3 else f"{n:g}") # type: ignore
from tqdm.auto import tqdm

def train_one_epoch_extra_losses(model, optimizer, loss_function, dataloader, device, epoch):
    model.train()

    total_size = 0
    running_loss = 0.0
    running_acc = 0.0
    epoch_loss = 0

    running_extra_losses = {}
    epoch_extra_losses = {} 

    progress_bar = tqdm(dataloader)
    progress_bar.set_description(f"Train")
    for data in progress_bar:
        images = data[0].to(device, dtype=torch.float32)
        targets = data[1].to(device, dtype=torch.int64)

        batch_size = images.shape[0]

        logits, extra_losses = model(images)
        loss = loss_function(logits, targets)
        for key, val in extra_losses.items():
            if key in running_extra_losses:
                running_extra_losses[key] += val
            else:
                running_extra_losses[key] = val

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        predicted = torch.argmax(logits, dim=1)
        acc = torch.sum(predicted == targets)

        running_loss += (loss.item() * batch_size)
        running_acc  += acc.item()
        total_size   += batch_size

        epoch_loss = running_loss / total_size
        epoch_acc = running_acc / total_size
        for key, val in running_extra_losses.items():
            epoch_extra_losses[key] = val.item() / total_size

        metrics = dict(Epoch=epoch, Train_Loss=epoch_loss, Train_Acc=epoch_acc,
                       LR=optimizer.param_groups[0]['lr'])
        metrics |= epoch_extra_losses
        progress_bar.set_postfix(metrics)

    return epoch_loss, epoch_acc, epoch_extra_losses

def train_one_epoch(model, optimizer, loss_function, dataloader, device, epoch):
    model.train()

    total_size = 0
    running_loss = 0.0
    running_acc = 0.0
    epoch_loss = 0

    progress_bar = tqdm(dataloader)
    progress_bar.set_description(f"Train")
    for data in progress_bar:
        images = data[0].to(device, dtype=torch.float32)
        targets = data[1].to(device, dtype=torch.int64)

        batch_size = images.shape[0]

        logits = model(images)
        loss = loss_function(logits, targets)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        predicted = torch.argmax(logits, dim=1)
        acc = torch.sum(predicted == targets)

        running_loss += (loss.item() * batch_size)
        running_acc  += acc.item()
        total_size   += batch_size

        epoch_loss = running_loss / total_size
        epoch_acc = running_acc / total_size

        progress_bar.set_postfix(Epoch=epoch, Train_Loss=epoch_loss, Train_Acc=epoch_acc,
                                 LR=optimizer.param_groups[0]['lr'])

    return epoch_loss, epoch_acc

@torch.inference_mode()
def valid_one_epoch(model, loss_function, dataloader, device, epoch):
    model.eval()

    total_size = 0
    running_loss = 0.0
    running_acc = 0.0
    epoch_loss = 0

    progress_bar = tqdm(dataloader)
    progress_bar.set_description(f"Validate")
    for data in progress_bar:
        images = data[0].to(device, dtype=torch.float)
        targets = data[1].to(device, dtype=torch.int64)

        batch_size = images.shape[0]

        logits = model(images)
        loss = loss_function(logits, targets)

        predicted = torch.argmax(logits, dim=1)
        acc = torch.sum(predicted == targets)

        running_loss += (loss.item() * batch_size)
        running_acc  += acc.item()
        total_size += batch_size

        epoch_loss = running_loss / total_size
        epoch_acc = running_acc / total_size

        progress_bar.set_postfix(Epoch=epoch, Valid_Loss=epoch_loss, Valid_Acc=epoch_acc)

    return epoch_loss, epoch_acc
