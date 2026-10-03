from pathlib import Path
import matplotlib.pyplot as plt

def plot_run(result, path):
    history = result["history"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    epochs = [r["epoch"] for r in history]

    for metric in ["train_loss", "val_loss"]:
        axes[0].plot(epochs, [r[metric] for r in history], label=metric)
    for metric in ["val_acc", "val_macro_f1"]:
        axes[1].plot(epochs, [r[metric] for r in history], label=metric)
    for metric in ["grad_norm", "grad_norm_max"]:
        axes[2].plot(epochs, [r[metric] for r in history], label=metric)

    for ax, ylabel in zip(axes, ["Loss", "Score", "Gradient norm before clip"]):
        ax.set(xlabel="Epoch", ylabel=ylabel)
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle(
        result["cfg"]["exp_id"] + " | " + str(result["cfg"])
    )
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)

def plot_compare(results, metric, path, title=""):
    fig, ax = plt.subplots(figsize=(8, 5))
    for result in results:
        history = result["history"]
        ax.plot(
            [r["epoch"] for r in history],
            [r[metric] for r in history],
            label=result["cfg"]["exp_id"],
        )
    ax.set(xlabel="Epoch", ylabel=metric, title=title)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
