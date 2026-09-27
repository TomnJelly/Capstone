#!/usr/bin/env python3
"""1채널 마스크 학습. 가중치는 weights/best.pt."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from common import JalilianDataset, UNet, bce_dice_loss, dice_score, find_data_root  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Vickers indent mask training")
    p.add_argument("--data", type=Path, default=None, help="Vicker_Database 폴더. 생략하면 자동으로 찾습니다.")
    p.add_argument("--out", type=Path, default=ROOT / "weights")
    p.add_argument("--size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--workers", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data = find_data_root(args.data)
    print(f"data={data}", flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        args.batch = min(args.batch, 4)
    args.out.mkdir(parents=True, exist_ok=True)
    use_amp = device.type == "cuda"

    train_ds = JalilianDataset(data, "I", size=args.size, augment=True)
    val_ds = JalilianDataset(data, "II", size=args.size, augment=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=args.workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=args.workers)

    model = UNet(base=32).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-5)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    print(
        f"device={device}  train={len(train_ds)}  val={len(val_ds)}  "
        f"size={args.size}  batch={args.batch}  lr={args.lr}",
        flush=True,
    )

    best = -1.0
    for epoch in range(1, args.epochs + 1):
        print(f"epoch {epoch:03d}/{args.epochs} start", flush=True)
        model.train()
        running = 0.0
        for batch in train_loader:
            x = batch["image"].to(device)
            y = batch["mask"].to(device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(x)
                loss = bce_dice_loss(logits, y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            running += float(loss.item()) * x.size(0)
        train_loss = running / max(len(train_ds), 1)
        sched.step()

        model.eval()
        dices = []
        with torch.no_grad():
            for batch in val_loader:
                x = batch["image"].to(device)
                y = batch["mask"].to(device)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    logits = model(x)
                dices.append(dice_score(logits, y))
        val_dice = sum(dices) / max(len(dices), 1)
        marker = ""
        if val_dice > best:
            best = val_dice
            torch.save(
                {
                    "model": model.state_dict(),
                    "size": args.size,
                    "epoch": epoch,
                    "val_dice": val_dice,
                    "arch": "unet1_mask256",
                },
                args.out / "best.pt",
            )
            marker = "  <- saved"
        print(
            f"epoch {epoch:03d}/{args.epochs}  loss={train_loss:.4f}  val_dice={val_dice:.4f}{marker}",
            flush=True,
        )

    print(f"best val_dice={best:.4f}", flush=True)
    print(f"weights: {args.out / 'best.pt'}", flush=True)


if __name__ == "__main__":
    main()
