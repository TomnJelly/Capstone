#!/usr/bin/env python3
"""v2 학습. DB-I / DB-II 각각 70% 학습, 30% 검증."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from common import (  # noqa: E402
    ARCH,
    DEFAULT_BASE,
    DEFAULT_SIZE,
    JalilianSubset,
    UNet,
    bce_dice_boundary_loss,
    db_dirs,
    diagonals_px,
    find_data_root,
    list_ids,
    load_gray,
    load_mask,
    predict_mask,
    save_splits,
    split_70_30,
    vertices_from_mask,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="v2 Vickers training, 70/30 per DB")
    p.add_argument("--data", type=Path, default=None)
    p.add_argument("--out", type=Path, default=ROOT / "weights")
    p.add_argument("--dbs", default="I,II", help="쉼표로 구분. 기본 I,II 각각 학습")
    p.add_argument("--size", type=int, default=DEFAULT_SIZE)
    p.add_argument("--base", type=int, default=DEFAULT_BASE)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--finetune-epochs", type=int, default=12)
    p.add_argument("--batch", type=int, default=6)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--patience", type=int, default=14)
    return p.parse_args()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def val_report(
    model: torch.nn.Module,
    data: Path,
    split: str,
    ids: list[str],
    size: int,
    device: torch.device,
    tta: bool,
) -> dict:
    model.eval()
    img_dir, mask_dir = db_dirs(data, split)
    dices: list[float] = []
    diags: list[float] = []
    hard: list[str] = []
    for stem in ids:
        gray = load_gray(img_dir / f"{stem}.png")
        gt = load_mask(mask_dir / f"{stem}.png")
        pred = predict_mask(model, gray, size, device, tta=tta, refine_crop=tta)
        p = pred > 127
        g = gt > 127
        den = int(p.sum() + g.sum())
        dice = float(2 * np.logical_and(p, g).sum() / den) if den else 0.0
        dices.append(dice)
        gv = vertices_from_mask(gt, gray)
        pv = vertices_from_mask(pred, gray)
        if gv is not None and pv is not None:
            _, _, dg = diagonals_px(gv)
            _, _, dp = diagonals_px(pv)
            if dg > 1.0:
                err = abs(dp - dg) / dg
                diags.append(err)
                if err >= 0.08 or dice < 0.85:
                    hard.append(stem)
            elif dice < 0.85:
                hard.append(stem)
        else:
            hard.append(stem)
    mean_dice = float(sum(dices) / max(len(dices), 1))
    mean_diag = float(sum(diags) / len(diags)) if diags else 1.0
    score = mean_dice - 0.6 * mean_diag
    return {
        "mean_dice": mean_dice,
        "mean_diag": mean_diag,
        "score": score,
        "hard": hard,
        "n": len(ids),
        "n_diag": len(diags),
    }


def train_epochs(
    model: torch.nn.Module,
    loader: DataLoader,
    val_ids: list[str],
    data: Path,
    split: str,
    opt: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    use_amp: bool,
    epochs: int,
    size: int,
    out_path: Path,
    meta: dict,
    patience: int,
    tta_val: bool,
) -> tuple[float, dict]:
    best_score = -1e9
    best_blob: dict | None = None
    wait = 0
    last_report: dict = {}
    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        seen = 0
        for batch in loader:
            x = batch["image"].to(device)
            y = batch["mask"].to(device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(x)
                loss = bce_dice_boundary_loss(logits, y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            running += float(loss.item()) * x.size(0)
            seen += x.size(0)
        train_loss = running / max(seen, 1)
        report = val_report(model, data, split, val_ids, size, device, tta=tta_val)
        last_report = report
        mark = ""
        if report["score"] > best_score:
            best_score = report["score"]
            wait = 0
            best_blob = {
                "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "size": size,
                "base": meta["base"],
                "epoch": epoch,
                "val_dice": report["mean_dice"],
                "val_diag": report["mean_diag"],
                "score": report["score"],
                "arch": ARCH,
                "db": split,
                **{k: meta[k] for k in meta if k not in {"base"}},
            }
            torch.save(best_blob, out_path)
            mark = "  <- saved"
        else:
            wait += 1
        print(
            f"  epoch {epoch:03d}/{epochs}  loss={train_loss:.4f}  "
            f"val_dice={report['mean_dice']:.4f}  val_diag={100 * report['mean_diag']:.2f}%  "
            f"score={report['score']:.4f}  hard={len(report['hard'])}{mark}",
            flush=True,
        )
        if wait >= patience:
            print(f"  early stop @ {epoch}", flush=True)
            break
    if best_blob is not None:
        model.load_state_dict(best_blob["model"])
        model.to(device)
    return best_score, last_report


def make_loader(
    ds: JalilianSubset,
    batch: int,
    workers: int,
    shuffle: bool,
    weights: list[float] | None = None,
) -> DataLoader:
    if weights is not None:
        sampler = WeightedRandomSampler(weights, num_samples=len(ds), replacement=True)
        return DataLoader(ds, batch_size=batch, sampler=sampler, num_workers=workers)
    return DataLoader(ds, batch_size=batch, shuffle=shuffle, num_workers=workers)


def train_one_db(args: argparse.Namespace, data: Path, db: str, device: torch.device) -> None:
    ids = list_ids(data, db)
    train_ids, val_ids = split_70_30(ids, seed=args.seed)
    save_splits(
        ROOT / "splits" / f"{db}.json",
        {"db": db, "seed": args.seed, "train": train_ids, "val": val_ids},
    )
    print(
        f"\n=== DB-{db}  train={len(train_ids)} ({100 * len(train_ids) / len(ids):.0f}%)  "
        f"val={len(val_ids)} ({100 * len(val_ids) / len(ids):.0f}%) ===",
        flush=True,
    )
    use_amp = device.type == "cuda"
    batch = args.batch if use_amp else min(args.batch, 2)
    train_ds = JalilianSubset(data, db, train_ids, size=args.size, augment=True)
    model = UNet(base=args.base).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    out_path = args.out / f"best_{db}.pt"
    meta = {"base": args.base, "seed": args.seed, "train_n": len(train_ids), "val_n": len(val_ids)}
    loader = make_loader(train_ds, batch, args.workers, shuffle=True)
    print("main training", flush=True)
    _, report = train_epochs(
        model,
        loader,
        val_ids,
        data,
        db,
        opt,
        scaler,
        device,
        use_amp,
        args.epochs,
        args.size,
        out_path,
        meta,
        args.patience,
        tta_val=False,
    )
    hard = []
    if args.finetune_epochs > 0:
        train_rep = val_report(model, data, db, train_ids, args.size, device, tta=False)
        hard = train_rep.get("hard", [])
        print(f"train hard examples={len(hard)}", flush=True)
    if hard and args.finetune_epochs > 0:
        print(f"hard-example finetune  n_hard={len(hard)}  extra={args.finetune_epochs}", flush=True)
        weights = [4.0 if stem in set(hard) else 1.0 for stem in train_ds.ids]
        ft_loader = make_loader(train_ds, batch, args.workers, shuffle=False, weights=weights)
        for g in opt.param_groups:
            g["lr"] = args.lr * 0.35
        train_epochs(
            model,
            ft_loader,
            val_ids,
            data,
            db,
            opt,
            scaler,
            device,
            use_amp,
            args.finetune_epochs,
            args.size,
            out_path,
            meta,
            max(6, args.patience // 2),
            tta_val=True,
        )
    blob = torch.load(out_path, map_location="cpu", weights_only=False)
    model.load_state_dict(blob["model"])
    model.to(device)
    final = val_report(model, data, db, val_ids, args.size, device, tta=True)
    summary_path = args.out / f"metrics_{db}.json"
    summary_path.write_text(json.dumps(final, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"DB-{db} done  dice={final['mean_dice']:.4f}  diag={100 * final['mean_diag']:.2f}%  "
        f"score={final['score']:.4f}  weights={out_path}",
        flush=True,
    )


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    data = find_data_root(args.data)
    print(f"data={data}", flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.out.mkdir(parents=True, exist_ok=True)
    print(
        f"device={device}  size={args.size}  base={args.base}  batch={args.batch}  "
        f"epochs={args.epochs}  lr={args.lr}",
        flush=True,
    )
    dbs = [x.strip().upper().replace("DB-", "") for x in args.dbs.split(",") if x.strip()]
    for db in dbs:
        train_one_db(args, data, db, device)
    print("v2 training finished", flush=True)


if __name__ == "__main__":
    main()
