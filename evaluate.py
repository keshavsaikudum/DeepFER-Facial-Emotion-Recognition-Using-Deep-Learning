"""Evaluate trained models on the held-out test set. Usage: python evaluate.py --data <dir> [--models cnn resnet18]"""
import argparse
import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader

from common import CLASSES, FERDataset, eval_transform, get_device, list_split, load_checkpoint


def predict(model, loader, device):
    probs = []
    with torch.no_grad():
        for x, _ in loader:
            probs.append(torch.softmax(model(x.to(device)), 1).cpu())
    return torch.cat(probs).numpy()


def latency_ms(model, size, device, runs=100):
    x = torch.randn(1, 1, size, size, device=device)
    with torch.no_grad():
        for _ in range(10):
            model(x)
        if device.type == "mps":
            torch.mps.synchronize()
        t = time.perf_counter()
        for _ in range(runs):
            model(x)
        if device.type == "mps":
            torch.mps.synchronize()
    return (time.perf_counter() - t) / runs * 1000


def plot_cm(cm, name, path):
    cmn = cm / cm.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(7), CLASSES, rotation=45, ha="right")
    ax.set_yticks(range(7), CLASSES)
    for i in range(7):
        for j in range(7):
            ax.text(j, i, f"{cmn[i, j]:.2f}", ha="center", va="center", color="white" if cmn[i, j] > .5 else "black", fontsize=8)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(f"{name}: normalised confusion matrix (test)")
    fig.tight_layout(); fig.savefig(path, dpi=140); plt.close(fig)


def plot_history(names, out):
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    for n in names:
        h = json.load(open(f"{out}/{n}_history.json"))
        e = [r["epoch"] for r in h]
        axs[0].plot(e, [r["train_loss"] for r in h], label=f"{n} train")
        axs[0].plot(e, [r["val_loss"] for r in h], "--", label=f"{n} val")
        axs[1].plot(e, [r["train_acc"] for r in h], label=f"{n} train")
        axs[1].plot(e, [r["val_acc"] for r in h], "--", label=f"{n} val")
    axs[0].set_title("Loss"); axs[1].set_title("Accuracy")
    for a in axs:
        a.set_xlabel("epoch"); a.legend(); a.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(f"{out}/training_curves.png", dpi=140); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--models", nargs="+", default=["cnn", "resnet18"])
    ap.add_argument("--out", default="runs")
    a = ap.parse_args()
    device = get_device()
    items = list_split(a.data, "test")
    y = np.array([l for _, l in items])
    summary, probs_all = {}, {}
    for name in a.models:
        model, size = load_checkpoint(f"{a.out}/{name}.pt", device)
        loader = DataLoader(FERDataset(items, eval_transform(size)), 256)
        p = predict(model, loader, device)
        probs_all[name] = p
        pred = p.argmax(1)
        rep = classification_report(y, pred, target_names=CLASSES, output_dict=True, digits=4)
        print(f"\n=== {name} ===\n" + classification_report(y, pred, target_names=CLASSES, digits=3))
        plot_cm(confusion_matrix(y, pred), name, f"{a.out}/{name}_confusion.png")
        summary[name] = dict(accuracy=rep["accuracy"], macro_precision=rep["macro avg"]["precision"],
                             macro_recall=rep["macro avg"]["recall"], macro_f1=rep["macro avg"]["f1-score"],
                             weighted_f1=rep["weighted avg"]["f1-score"], latency_ms=latency_ms(model, size, device),
                             params=sum(p.numel() for p in model.parameters()),
                             per_class={c: rep[c] for c in CLASSES})
    if len(a.models) > 1:  # simple ensemble: average softmax probabilities
        pred = sum(probs_all.values()).argmax(1)
        rep = classification_report(y, pred, target_names=CLASSES, output_dict=True)
        summary["ensemble"] = dict(accuracy=rep["accuracy"], macro_precision=rep["macro avg"]["precision"],
                                   macro_recall=rep["macro avg"]["recall"], macro_f1=rep["macro avg"]["f1-score"],
                                   weighted_f1=rep["weighted avg"]["f1-score"])
    plot_history(a.models, a.out)
    json.dump(summary, open(f"{a.out}/metrics.json", "w"), indent=1)
    print("\nSUMMARY")
    for k, v in summary.items():
        print(f"{k:9s} acc {v['accuracy']:.4f}  macroP {v['macro_precision']:.4f}  macroR {v['macro_recall']:.4f}  macroF1 {v['macro_f1']:.4f}"
              + (f"  {v['latency_ms']:.2f} ms/img" if "latency_ms" in v else ""))


if __name__ == "__main__":
    main()
