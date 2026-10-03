import time
import numpy as np
import torch

from model import MLP, count_params, EXPECTED_PARAMS
from data import iterate_batches
from optimizer import build_optimizer, clip_gradients
from train import DEFAULT_CFG, set_seed, compute_loss, evaluate

def run_experiment(cfg, data):
    cfg = {**DEFAULT_CFG, **cfg}
    if cfg["lr"] is None or cfg["lr"] <= 0:
        raise ValueError("Learning rate must be positive")

    device = data["X_tr"].device
    is_cuda = device.type == "cuda"
    precision = cfg["precision"]

    if precision not in {"fp32", "fp16", "bf16"}:
        raise ValueError(precision)
    if precision != "fp32" and not is_cuda:
        raise ValueError("Mixed precision experiments require CUDA")
    if precision == "bf16" and not torch.cuda.is_bf16_supported():
        raise ValueError("GPU này không hỗ trợ BF16 native")

    amp_enabled = precision != "fp32"
    amp_dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    scaler = torch.amp.GradScaler(
        "cuda", enabled=(precision == "fp16")
    )

    def synchronize():
        if is_cuda:
            torch.cuda.synchronize(device)

    set_seed(cfg["seed"])
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

    synchronize()
    if is_cuda:
        torch.cuda.reset_peak_memory_stats(device)
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
        clipped_steps = 0
        skipped_steps = 0
        attempted_steps = 0

        for xb, yb in iterate_batches(
            data["X_tr"], data["y_tr"],
            cfg["batch"], generator=generator,
        ):
            attempted_steps += 1
            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=amp_enabled,
            ):
                logits = model(xb)
                # Tính loss bằng FP32 để ổn định số học.
                loss = compute_loss(logits.float(), yb, cfg["loss"])

            if not torch.isfinite(loss).item():
                diverged = True
                break

            scaler.scale(loss).backward()
            # Gradient phải được unscale trước khi đo hoặc clip.
            scaler.unscale_(optimizer)

            grad_norm = clip_gradients(
                model.parameters(), cfg["clip_norm"]
            )

            if not np.isfinite(grad_norm):
                if precision == "fp16":
                    # GradScaler bỏ cập nhật khi gradient overflow.
                    scaler.step(optimizer)
                    scaler.update()
                    skipped_steps += 1
                    continue

                diverged = True
                break

            gradient_norms.append(grad_norm)
            if (cfg["clip_norm"] is not None
                    and grad_norm > cfg["clip_norm"]):
                clipped_steps += 1

            scaler.step(optimizer)
            scaler.update()

        synchronize()
        train_seconds = time.perf_counter() - epoch_started

        if diverged:
            print(f"{cfg['exp_id']}: diverged at epoch {epoch}")
            break
        if not gradient_norms:
            diverged = True
            print(f"{cfg['exp_id']}: no finite gradient updates")
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
            clip_fraction=clipped_steps / len(gradient_norms),
            skipped_steps=skipped_steps,
            attempted_steps=attempted_steps,
            train_seconds=train_seconds,
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
            f"clip={row['clip_fraction']:.1%} "
            f"skip={skipped_steps} "
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
