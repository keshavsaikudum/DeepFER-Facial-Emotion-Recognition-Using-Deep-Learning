"""Train DeepFER models. Usage: python train.py --data <dir with train/ test/> --model cnn|resnet18"""
import argparse
import json
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, WeightedRandomSampler

from common import (CLASSES, FERDataset, build, eval_transform, get_device, list_split, seed_everything,
                    train_transform)


def run_epoch(model, loader, device, loss_fn, opt=None, sched=None):
    train = opt is not None
    model.train(train)
    tot_loss = correct = n = 0
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = loss_fn(out, y)
            if train:
                opt.zero_grad()
                loss.backward()
                opt.step()
                if sched:
                    sched.step()
            tot_loss += loss.item() * len(y)
            correct += (out.argmax(1) == y).sum().item()
            n += len(y)
    return tot_loss / n, correct / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default="cnn", choices=["cnn", "resnet18"])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--freeze-epochs", type=int, default=3, help="resnet18 only: epochs training just the head")
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", default="runs")
    ap.add_argument("--limit", type=int, default=0, help="debug: cap train images")
    a = ap.parse_args()

    seed_everything()
    device = get_device()
    model, size = build(a.model)
    model.to(device)

    items = list_split(a.data, "train")
    labels = [y for _, y in items]
    tr_items, va_items = train_test_split(items, test_size=0.1, stratify=labels, random_state=42)
    if a.limit:
        tr_items = tr_items[:a.limit]
    # class-balanced sampling: 'disgust' has ~10x fewer images than the others
    tr_labels = np.array([y for _, y in tr_items])
    w = 1.0 / np.bincount(tr_labels, minlength=len(CLASSES)) ** 0.5  # sqrt-inverse: balance without heavy oversampling
    sampler = WeightedRandomSampler(w[tr_labels], len(tr_items), replacement=True)

    tr = DataLoader(FERDataset(tr_items, train_transform(size)), a.bs, sampler=sampler, num_workers=0)
    va = DataLoader(FERDataset(va_items, eval_transform(size)), 256, num_workers=0)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

    def make_opt(params, lr, total_epochs):
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
        sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=total_epochs * len(tr))
        return opt, sch

    stages = [(a.epochs, a.lr)]
    if a.model == "resnet18":  # stage 1: head only, stage 2: fine-tune everything at lower LR
        stages = [(a.freeze_epochs, a.lr), (a.epochs - a.freeze_epochs, a.lr / 5)]

    hist, best, ep = [], 0.0, 0
    import os
    os.makedirs(a.out, exist_ok=True)
    ckpt = f"{a.out}/{a.model}.pt"
    for si, (n_ep, lr) in enumerate(stages):
        if a.model == "resnet18":
            model.set_backbone_frozen(si == 0)
        opt, sch = make_opt([p for p in model.parameters() if p.requires_grad], lr, n_ep)
        for _ in range(n_ep):
            ep += 1
            t0 = time.time()
            tl, ta = run_epoch(model, tr, device, loss_fn, opt, sch)
            vl, vacc = run_epoch(model, va, device, loss_fn)
            hist.append(dict(epoch=ep, train_loss=tl, train_acc=ta, val_loss=vl, val_acc=vacc))
            if vacc > best:
                best = vacc
                torch.save({"model": a.model, "state": model.state_dict(), "classes": CLASSES}, ckpt)
            print(f"[{a.model}] ep {ep:02d} train {tl:.3f}/{ta:.3f} val {vl:.3f}/{vacc:.3f} best {best:.3f} ({time.time()-t0:.0f}s)", flush=True)
            json.dump(hist, open(f"{a.out}/{a.model}_history.json", "w"), indent=1)


if __name__ == "__main__":
    main()
