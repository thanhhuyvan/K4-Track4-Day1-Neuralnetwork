import random
import time
import numpy as np
import torch
import torch.nn.functional as F

from data import iterate_batches
from model import MLP, count_params, EXPECTED_PARAMS
from optimizer import build_optimizer, clip_gradients

DEFAULT_CFG = dict(
    exp_id="base-s1",
    group="baseline",
    description="M-base, SGD momentum",
    loss="ce",
    optimizer="sgd_momentum",
    lr=None,
    weight_decay=0.0,
    momentum=0.9,
    batch=512,
    epochs=20,
    hidden=(256, 128),
    dropout=0.0,
    init="he",
    clip_norm=None,
    precision="fp32",
    seed=1,
)

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def compute_loss(logits, y, loss_name):
    if loss_name == "ce":
        return F.cross_entropy(logits, y)
    if loss_name == "mse":
        target = F.one_hot(y, num_classes=7).float()
        return F.mse_loss(logits.softmax(dim=1), target)
    raise ValueError(loss_name)

def macro_f1_from_confusion(cm):
    cm = np.asarray(cm, dtype=np.float64)
    tp = np.diag(cm)
    denominator = cm.sum(0) + cm.sum(1)
    f1 = np.divide(
        2 * tp, denominator,
        out=np.zeros_like(tp), where=denominator > 0
    )
    return float(f1.mean())

@torch.no_grad()
def evaluate(model, X, y, loss_name="ce", batch_size=8192):
    original_mode = model.training
    model.eval()
    loss_sum = 0.0
    cm = torch.zeros((7, 7), dtype=torch.int64, device=X.device)

    try:
        for start in range(0, len(y), batch_size):
            xb = X[start:start + batch_size]
            yb = y[start:start + batch_size]
            logits = model(xb)
            loss_sum += compute_loss(logits, yb, loss_name).item() * len(yb)
            pred = logits.argmax(1)
            cm += torch.bincount(
                yb * 7 + pred, minlength=49
            ).reshape(7, 7)

        cm_np = cm.cpu().numpy()
        return dict(
            loss=loss_sum / len(y),
            acc=float(np.trace(cm_np) / cm_np.sum()),
            macro_f1=macro_f1_from_confusion(cm_np),
        )
    finally:
        model.train(original_mode)

def run_experiment(cfg, data):
    cfg = {**DEFAULT_CFG, **cfg}
    if cfg["lr"] is None or cfg["lr"] <= 0:
        raise ValueError("Choose a positive learning rate")
    if cfg["precision"] != "fp32":
        raise ValueError("Part 2 currently supports fp32 only")

    set_seed(cfg["seed"])
    device = data["X_tr"].device
    model = MLP(
        hidden=tuple(cfg["hidden"]),
        dropout=cfg["dropout"],
        init=cfg["init"],
    ).to(device)
    assert count_params(model) == EXPECTED_PARAMS[tuple(cfg["hidden"])]

    optimizer = build_optimizer(
        cfg["optimizer"], model.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
        momentum=cfg["momentum"],
    )
    generator = torch.Generator(device=device)
    generator.manual_seed(cfg["seed"])

    is_cuda = device.type == "cuda"

    def synchronize():
        if is_cuda:
            torch.cuda.synchronize(device)

    if is_cuda:
        torch.cuda.reset_peak_memory_stats(device)

    synchronize()
    started = time.perf_counter()
    initial = evaluate(
        model, data["X_val"], data["y_val"], cfg["loss"]
    )
    history = []
    best_loss = float("inf")
    best_state = None
    best_metrics = None
    best_epoch = None
    diverged = False

    for epoch in range(1, cfg["epochs"] + 1):
        synchronize()
        epoch_started = time.perf_counter()
        model.train()
        gradient_norms = []

        for xb, yb in iterate_batches(
            data["X_tr"], data["y_tr"],
            cfg["batch"], generator=generator,
        ):
            optimizer.zero_grad(set_to_none=True)
            loss = compute_loss(model(xb), yb, cfg["loss"])

            if not torch.isfinite(loss).item():
                diverged = True
                break

            loss.backward()
            grad_norm = clip_gradients(
                model.parameters(), cfg["clip_norm"]
            )
            if not np.isfinite(grad_norm):
                diverged = True
                break

            gradient_norms.append(grad_norm)
            optimizer.step()

        if diverged:
            print(f"{cfg['exp_id']}: diverged at epoch {epoch}")
            break

        train_metrics = evaluate(
            model, data["X_tr"], data["y_tr"], cfg["loss"]
        )
        val_metrics = evaluate(
            model, data["X_val"], data["y_val"], cfg["loss"]
        )

        if not all(np.isfinite(m["loss"])
                   for m in (train_metrics, val_metrics)):
            diverged = True
            break

        synchronize()
        row = dict(
            epoch=epoch,
            train_loss=train_metrics["loss"],
            val_loss=val_metrics["loss"],
            val_acc=val_metrics["acc"],
            val_macro_f1=val_metrics["macro_f1"],
            grad_norm=float(np.mean(gradient_norms)),
            grad_norm_max=float(np.max(gradient_norms)),
            seconds=time.perf_counter() - epoch_started,
        )
        history.append(row)

        if val_metrics["loss"] < best_loss:
            best_loss = val_metrics["loss"]
            best_epoch = epoch
            best_metrics = dict(val_metrics)
            best_state = {
                name: tensor.detach().cpu().clone()
                for name, tensor in model.state_dict().items()
            }

        print(
            f"{cfg['exp_id']} | {epoch:02d}/{cfg['epochs']} | "
            f"train={row['train_loss']:.4f} "
            f"val={row['val_loss']:.4f} "
            f"acc={row['val_acc']:.4f} "
            f"F1={row['val_macro_f1']:.4f} "
            f"grad={row['grad_norm']:.3f} "
            f"time={row['seconds']:.1f}s",
            flush=True,
        )

    synchronize()
    return dict(
        cfg=cfg,
        history=history,
        initial_val_loss=initial["loss"],
        best_epoch=best_epoch,
        best_val_loss=None if best_metrics is None else best_metrics["loss"],
        val_acc=None if best_metrics is None else best_metrics["acc"],
        val_macro_f1=None if best_metrics is None else best_metrics["macro_f1"],
        total_seconds=time.perf_counter() - started,
        peak_memory_mb=(
            torch.cuda.max_memory_allocated(device) / 1024**2
            if is_cuda else None
        ),
        diverged=diverged,
        best_state=best_state,
    )
