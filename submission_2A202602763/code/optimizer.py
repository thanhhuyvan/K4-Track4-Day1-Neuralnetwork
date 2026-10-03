import torch

def build_optimizer(name, params, lr, weight_decay=0.0, momentum=0.9):
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, weight_decay=weight_decay)
    if name == "sgd_momentum":
        return torch.optim.SGD(
            params, lr=lr, momentum=momentum, weight_decay=weight_decay
        )
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    raise ValueError(f"Unknown optimizer: {name}")

def clip_gradients(params, max_norm=None):
    params = list(params)
    if max_norm is not None:
        # PyTorch trả về norm TRƯỚC khi clip.
        return float(torch.nn.utils.clip_grad_norm_(params, max_norm).item())

    norms = [p.grad.detach().float().norm(2)
             for p in params if p.grad is not None]
    if not norms:
        return 0.0
    return float(torch.stack(norms).norm(2).item())
