import torch
from env_variables import BATCH_SIZE

def validate(model, data):  # accuracy
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    total = 0
    correct = 0
    for i, (images, labels) in enumerate(data):
        labels = labels[0, :]
        if len(labels) != BATCH_SIZE:
            break
        labels = labels.to(device, dtype=torch.long)
        images = images[0, :, :, :]
        images = images.to(device, dtype=torch.float)
        x = model(images)
        value, pred = torch.max(x, 1)
        # pred = pred.data.cpu()
        total += float(x.size(0))
        correct += float(torch.sum(pred == labels))
    return correct * 100 / total


def cel(model, data, ce):  # cross-entropy
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    results = []
    with(torch.set_grad_enabled(False)):
        for i, (images, labels) in enumerate(data):
            labels = labels[0, :]
            labels = labels.to(device, dtype=torch.long)
            images = images[0, :, :, :]
            images = images.to(device, dtype=torch.float)
            pred = model(images)
            results.append(ce(pred, labels))
    return sum(results) / len(results)
