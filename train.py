#!/usr/bin/env python3
"""마스크 + 4코너 히트맵 학습. 가중치는 weights/best.pt."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from common import (  # noqa: E402
    JalilianDataset,
    UNet,
    dice_score,
    mean_diag_error,
    multitask_loss,
    optimizer_step,
    pick_accelerator,
    save_ckpt,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Vickers indent mask+corner training")
    p.add_argument("--data", type=Path, default=Path(r"C:\Users\seoyu\Downloads\Vicker_Database"))
    p.add_argument("--out", type=Path, default=ROOT / "weights")
    p.add_argument("--size", type=int, default=384)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--workers", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device, kind, xm = pick_accelerator()
    if kind not in ("tpu", "cuda"):
        args.batch = min(args.batch, 4)
    args.out.mkdir(parents=True, exist_ok=True)
    use_amp = kind == "cuda"

    train_ds = JalilianDataset(args.data, "I", size=args.size, augment=True)
    val_ds = JalilianDataset(args.data, "II", size=args.size, augment=False)
    print(f"expert verts train={len(train_ds.expert)}/{len(train_ds)}  val={len(val_ds.expert)}/{len(val_ds)}", flush=True)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=args.workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=args.workers)

    model = UNet(base=32, out_ch=5).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    warmup = torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.1, total_iters=3)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(args.epochs - 3, 1), eta_min=1e-5)
    sched = torch.optim.lr_scheduler.SequentialLR(opt, [warmup, cosine], milestones=[3])
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    print(
        f"device={device}  kind={kind}  train={len(train_ds)}  val={len(val_ds)}  "
        f"size={args.size}  batch={args.batch}  lr={args.lr}",
        flush=True,
    )

    best_dice = -1.0
    best_diag = 1e9
    for epoch in range(1, args.epochs + 1):
        print(f"epoch {epoch:03d}/{args.epochs} start", flush=True)
        model.train()
        running = 0.0
        train_dices = []
        for batch in train_loader:
            x = batch["image"].to(device)
            y = batch["mask"].to(device)
            heat = batch["heat"].to(device)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(x)
                loss = multitask_loss(logits, y, heat)
            optimizer_step(loss, opt, model, kind, xm, scaler)
            running += float(loss.item()) * x.size(0)
            train_dices.append(dice_score(logits.detach(), y))
        train_loss = running / max(len(train_ds), 1)
        train_dice = sum(train_dices) / max(len(train_dices), 1)
        sched.step()

        model.eval()
        dices = []
        diag_errs = []
        with torch.no_grad():
            for batch in val_loader:
                x = batch["image"].to(device)
                y = batch["mask"].to(device)
                with torch.cuda.amp.autocast(enabled=use_amp):
                    logits = model(x)
                dices.append(dice_score(logits, y))
                err = mean_diag_error(logits, batch["verts"])
                if err is not None:
                    diag_errs.append(err)
        val_dice = sum(dices) / max(len(dices), 1)
        val_diag = sum(diag_errs) / max(len(diag_errs), 1) if diag_errs else None
        marker = ""
        should_save = False
        if val_diag is not None and val_dice >= 0.85 and val_diag < best_diag:
            best_diag = val_diag
            should_save = True
        elif best_diag >= 1e8 and val_dice > best_dice:
            should_save = True
        if val_dice > best_dice:
            best_dice = val_dice
        if should_save:
            save_ckpt(
                args.out / "best.pt",
                model,
                {
                    "size": args.size,
                    "epoch": epoch,
                    "val_dice": val_dice,
                    "val_diag": val_diag,
                    "arch": "unet4_heat5",
                },
                kind,
                xm,
            )
            marker = "  <- saved"
        lr_now = opt.param_groups[0]["lr"]
        diag_txt = f"{100 * val_diag:.2f}%" if val_diag is not None else "n/a"
        print(
            f"epoch {epoch:03d}/{args.epochs}  loss={train_loss:.4f}  train_dice={train_dice:.4f}  "
            f"val_dice={val_dice:.4f}  val_diag={diag_txt}  lr={lr_now:.2e}{marker}",
            flush=True,
        )

    print(f"best val_dice={best_dice:.4f}  best val_diag={100 * best_diag:.2f}%", flush=True)
    print(f"weights: {args.out / 'best.pt'}", flush=True)
    print(f"미리보기: python preview.py --ckpt {args.out / 'best.pt'}", flush=True)


if __name__ == "__main__":
    main()
