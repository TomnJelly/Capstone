#!/usr/bin/env python3
"""
Kaggle 학습 전용. 이 파일 전체를 노트북 한 셀에 붙여 넣으면 best.pt 가 나옵니다.
1채널 마스크 U-Net. 꼭짓점은 마스크 끝점(N/E/S/W)에서 뽑습니다.
학습이 끝나면 검증 8장을 셀 로그에 띄웁니다.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

EPOCHS = 30
SIZE = 256
LR = 3e-4
PREVIEW_N = 8
Vertex = tuple[float, float]
CORNERS = ("N", "E", "S", "W")


def find_data_root() -> Path:
    kaggle = Path("/kaggle/input")
    if kaggle.exists():
        for mask_dir in kaggle.rglob("exact-masks-I"):
            data_root = mask_dir.parent
            if (data_root / "DB-I").exists():
                return data_root
        raise FileNotFoundError("/kaggle/input 아래에 DB-I + exact-masks-I 가 없습니다.")
    local = Path(r"C:\Users\seoyu\Downloads\Vicker_Database")
    if local.exists():
        return local
    raise FileNotFoundError("Jalilian 데이터 경로를 찾지 못했습니다.")


def db_dirs(root: Path, split: str) -> tuple[Path, Path]:
    tag = split.upper().replace("DB-", "")
    return root / f"DB-{tag}", root / f"exact-masks-{tag}"


def list_ids(root: Path, split: str) -> list[str]:
    img_dir, mask_dir = db_dirs(root, split)
    return [p.stem for p in sorted(img_dir.glob("*.png")) if (mask_dir / p.name).exists()]


def load_gray(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    return img


def load_mask(path: Path) -> np.ndarray:
    mask = load_gray(path)
    return np.where(mask > 127, 255, 0).astype(np.uint8)


class JalilianDataset(Dataset):
    def __init__(self, root: Path, split: str, size: int = 256, augment: bool = False) -> None:
        self.img_dir, self.mask_dir = db_dirs(root, split)
        self.size = size
        self.augment = augment
        self.ids = list_ids(root, split)
        if not self.ids:
            raise FileNotFoundError(f"이미지/마스크 없음: {self.img_dir} + {self.mask_dir}")

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int) -> dict:
        stem = self.ids[idx]
        image = load_gray(self.img_dir / f"{stem}.png")
        mask = load_mask(self.mask_dir / f"{stem}.png")
        image = cv2.resize(image, (self.size, self.size), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, (self.size, self.size), interpolation=cv2.INTER_NEAREST)
        if self.augment:
            if np.random.rand() < 0.5:
                image, mask = cv2.flip(image, 1), cv2.flip(mask, 1)
            if np.random.rand() < 0.5:
                image, mask = cv2.flip(image, 0), cv2.flip(mask, 0)
            k = int(np.random.randint(0, 4))
            if k:
                image, mask = np.rot90(image, k).copy(), np.rot90(mask, k).copy()
            alpha = 0.75 + 0.5 * np.random.rand()
            beta = np.random.randint(-18, 19)
            image = np.clip(image.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
        return {
            "image": torch.from_numpy(image.astype(np.float32) / 255.0).unsqueeze(0),
            "mask": torch.from_numpy((mask > 127).astype(np.float32)).unsqueeze(0),
            "id": stem,
        }


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UNet(nn.Module):
    def __init__(self, base: int = 32) -> None:
        super().__init__()
        self.enc1 = ConvBlock(1, base)
        self.enc2 = ConvBlock(base, base * 2)
        self.enc3 = ConvBlock(base * 2, base * 4)
        self.enc4 = ConvBlock(base * 4, base * 8)
        self.pool = nn.MaxPool2d(2)
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec3 = ConvBlock(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = ConvBlock(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = ConvBlock(base * 2, base)
        self.out = nn.Conv2d(base, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up3(e4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.out(d1)


def bce_dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    bce = nn.functional.binary_cross_entropy_with_logits(logits, target)
    pred = torch.sigmoid(logits)
    num = 2 * (pred * target).sum(dim=(1, 2, 3))
    den = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + eps
    return bce + (1 - (num / den).mean())


@torch.no_grad()
def dice_score(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> float:
    pred = (torch.sigmoid(logits.float()) > 0.5).float()
    num = 2 * (pred * target).sum(dim=(1, 2, 3))
    den = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + eps
    return float((num / den).mean().item())


@torch.no_grad()
def predict_mask(model: nn.Module, gray: np.ndarray, size: int, device: torch.device) -> np.ndarray:
    h, w = gray.shape[:2]
    inp = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
    tensor = torch.from_numpy(inp.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0).to(device)
    prob = torch.sigmoid(model(tensor).float())[0, 0].cpu().numpy()
    pred = (prob > 0.5).astype(np.uint8) * 255
    return cv2.resize(pred, (w, h), interpolation=cv2.INTER_NEAREST)


def vertices_from_mask(mask: np.ndarray) -> dict[str, Vertex] | None:
    binary = (mask > 127).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea)
    if cv2.contourArea(cnt) < 20:
        return None
    pts = cnt.reshape(-1, 2)
    return {
        "N": (float(pts[np.argmin(pts[:, 1]), 0]), float(pts[np.argmin(pts[:, 1]), 1])),
        "E": (float(pts[np.argmax(pts[:, 0]), 0]), float(pts[np.argmax(pts[:, 0]), 1])),
        "S": (float(pts[np.argmax(pts[:, 1]), 0]), float(pts[np.argmax(pts[:, 1]), 1])),
        "W": (float(pts[np.argmin(pts[:, 0]), 0]), float(pts[np.argmin(pts[:, 0]), 1])),
    }


def diagonals_px(verts: dict[str, Vertex]) -> tuple[float, float, float]:
    d1 = float(np.hypot(verts["S"][0] - verts["N"][0], verts["S"][1] - verts["N"][1]))
    d2 = float(np.hypot(verts["E"][0] - verts["W"][0], verts["E"][1] - verts["W"][1]))
    return d1, d2, 0.5 * (d1 + d2)


def overlay_mask(gray: np.ndarray, mask: np.ndarray, color: tuple[int, int, int] = (46, 204, 113)) -> np.ndarray:
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    tint = np.zeros_like(vis)
    tint[:] = color
    alpha = (mask > 127).astype(np.float32) * 0.45
    out = vis.astype(np.float32) * (1 - alpha[..., None]) + tint.astype(np.float32) * alpha[..., None]
    return np.clip(out, 0, 255).astype(np.uint8)


def draw_geometry(gray: np.ndarray, verts: dict[str, Vertex] | None, color: tuple[int, int, int]) -> np.ndarray:
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR) if gray.ndim == 2 else gray.copy()
    if verts is None:
        return vis
    pts = [tuple(map(int, verts[k])) for k in CORNERS]
    cv2.line(vis, pts[0], pts[2], color, 2, cv2.LINE_AA)
    cv2.line(vis, pts[1], pts[3], color, 2, cv2.LINE_AA)
    for key, pt in zip(CORNERS, pts):
        cv2.circle(vis, pt, 8, color, -1, cv2.LINE_AA)
        cv2.putText(vis, key, (pt[0] + 10, pt[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    return vis


def _show_bgr(title: str, bgr: np.ndarray) -> None:
    h, w = bgr.shape[:2]
    max_w = 1100
    if w > max_w:
        s = max_w / w
        bgr = cv2.resize(bgr, (max_w, int(h * s)), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(11, 5.2))
        ax.imshow(rgb)
        ax.set_title(title)
        ax.axis("off")
        plt.show()
        plt.close(fig)
    except Exception:
        print(title, flush=True)


@torch.no_grad()
def write_previews(model: nn.Module, data: Path, device: torch.device) -> None:
    img_dir, mask_dir = db_dirs(data, "II")
    ids = list_ids(data, "II")[:PREVIEW_N]
    model.eval()
    print(f"preview {len(ids)}장  (왼쪽=정답, 오른쪽=예측)", flush=True)
    for stem in ids:
        gray = load_gray(img_dir / f"{stem}.png")
        gt = load_mask(mask_dir / f"{stem}.png")
        pred = predict_mask(model, gray, SIZE, device)
        gt_verts = vertices_from_mask(gt)
        pred_verts = vertices_from_mask(pred)
        left = draw_geometry(overlay_mask(gray, gt), gt_verts, (46, 204, 113))
        right = draw_geometry(overlay_mask(gray, pred, (230, 126, 34)), pred_verts, (230, 126, 34))
        gap = np.full((left.shape[0], 8, 3), 40, dtype=np.uint8)
        title = stem
        if gt_verts is not None and pred_verts is not None:
            _, _, dg = diagonals_px(gt_verts)
            _, _, dp = diagonals_px(pred_verts)
            if dg > 1.0:
                title = f"{stem}  diag={100 * abs(dp - dg) / dg:.1f}%"
        _show_bgr(title, np.hstack([left, gap, right]))


def main() -> None:
    data = find_data_root()
    out = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path("weights")
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    batch = 8 if use_amp else 4
    print("data =", data)
    print("device =", device, torch.cuda.get_device_name(0) if use_amp else "cpu")
    train_ds = JalilianDataset(data, "I", size=SIZE, augment=True)
    val_ds = JalilianDataset(data, "II", size=SIZE, augment=False)
    nw = 2 if use_amp else 0
    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True, num_workers=nw)
    val_loader = DataLoader(val_ds, batch_size=batch, shuffle=False, num_workers=nw)
    model = UNet().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=1e-5)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    print(f"train={len(train_ds)} val={len(val_ds)} batch={batch} epochs={EPOCHS} size={SIZE} lr={LR}")
    best = -1.0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        running = 0.0
        for batch_data in train_loader:
            x = batch_data["image"].to(device)
            y = batch_data["mask"].to(device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(x)
                loss = bce_dice_loss(logits, y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            running += float(loss.item()) * x.size(0)
        train_loss = running / max(len(train_ds), 1)
        sched.step()
        model.eval()
        dices = []
        with torch.no_grad():
            for batch_data in val_loader:
                x = batch_data["image"].to(device)
                y = batch_data["mask"].to(device)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    logits = model(x)
                dices.append(dice_score(logits, y))
        val_dice = sum(dices) / max(len(dices), 1)
        mark = ""
        if val_dice > best:
            best = val_dice
            torch.save(
                {
                    "model": {k: v.detach().to("cpu") for k, v in model.state_dict().items()},
                    "size": SIZE,
                    "epoch": epoch,
                    "val_dice": val_dice,
                    "arch": "unet1_mask256",
                },
                out / "best.pt",
            )
            mark = "  <- saved"
        print(f"epoch {epoch:03d}/{EPOCHS}  loss={train_loss:.4f}  val_dice={val_dice:.4f}{mark}", flush=True)
    print("best val_dice =", round(best, 4))
    print("weights =", out / "best.pt")
    ckpt = out / "best.pt"
    if ckpt.exists():
        blob = torch.load(ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(blob["model"])
        model.to(device)
    write_previews(model, data, device)


main()
