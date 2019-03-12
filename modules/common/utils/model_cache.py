import torch


def save_model(epoch, model, optimizer, loss, path):
    print('Saving model...')
    torch.save({
        'epoch': epoch,
        'model': model,
        'optimizer': optimizer,
        'loss': loss
    }, path)


def load_model(path):
    if torch.cuda.is_available():
        checkpoint = torch.load(path)
    else:
        checkpoint = torch.load(path, map_location='cpu')
    model = checkpoint['model']
    optimizer = checkpoint['optimizer']

    return model, optimizer
