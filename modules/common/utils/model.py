import torch


def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


def clip_gradient_norm(model, clip):
    """clip_grad_norm helps prevent the exploding gradient problem in RNNs / LSTMs."""
    torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
    for p in model.parameters():
        p.data.add_(-learning_rate, p.grad.data)
