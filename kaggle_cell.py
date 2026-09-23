#!/usr/bin/env python3
"""
Kaggle 학습용. 이 파일 전체를 노트북 한 셀에 붙여 넣으세요.
먼저 Vicker_Database 를 Add Input 해야 합니다. (DB-I, exact-masks-I, measure_lines_serie1)
"""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

EPOCHS = 40
SIZE = 384
LR = 3e-4


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

Vertex = tuple[float, float]
CORNERS = ("N", "E", "S", "W")


def pick_accelerator() -> tuple[torch.device, str, object | None]:
    """TPU(torch_xla) → GPU → CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda"), "cuda", None
    tpu_env = (
        os.environ.get("PJRT_DEVICE", "").upper() == "TPU"
        or bool(os.environ.get("TPU_NAME"))
        or Path("/dev/accel0").exists()
    )
    if tpu_env:
        os.environ.setdefault("PJRT_DEVICE", "TPU")
        try:
            import torch_xla.core.xla_model as xm  # type: ignore
        except ImportError as err:
            raise SystemExit(
                "이 노트북은 TPU입니다. PyTorch는 torch_xla가 필요합니다.\n"
                "맨 위 셀에서 `%pip install -q torch_xla` 를 실행하세요."
            ) from err
        return xm.xla_device(), "tpu", xm
    return torch.device("cpu"), "cpu", None


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


def preprocess_for_model(gray: np.ndarray) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def _mid(a: tuple[float, float], b: tuple[float, float]) -> Vertex:
    return ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)


def load_expert_vertices(root: Path, split: str) -> dict[str, dict[str, Vertex]]:
    """시리즈 1 측정선(measure_lines)의 네 변 중점으로 N/E/S/W."""
    available = set(list_ids(root, split))
    tag = split.upper().replace("DB-", "")
    dbn = "1" if tag == "I" else "2"
    path = Path(root) / "04_Algorithmus_Statistik" / "ground_truth" / f"db{dbn}_measure_lines_serie1.txt"
    out: dict[str, dict[str, Vertex]] = {}
    if not path.exists():
        return out
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        name, rest = line.split(":", 1)
        stem = Path(name.strip()).stem
        if stem not in available:
            continue
        pts: list[tuple[float, float]] = []
        for tok in rest.split():
            if "/" not in tok:
                continue
            xs, ys = tok.split("/", 1)
            try:
                pts.append((float(xs), float(ys)))
            except ValueError:
                continue
        if len(pts) != 4:
            continue
        tl, bl, tr, br = pts
        out[stem] = {
            "N": _mid(tl, tr),
            "E": _mid(tr, br),
            "S": _mid(bl, br),
            "W": _mid(tl, bl),
        }
    return out


def make_heatmaps(h: int, w: int, verts: dict[str, Vertex], sigma: float) -> np.ndarray:
    yy, xx = np.mgrid[0:h, 0:w]
    heat = np.zeros((4, h, w), dtype=np.float32)
    sig2 = 2.0 * max(sigma, 1.5) ** 2
    for i, key in enumerate(CORNERS):
        x, y = verts[key]
        heat[i] = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / sig2)
    return heat


def peaks_from_heatmaps(heat: np.ndarray) -> dict[str, Vertex]:
    verts: dict[str, Vertex] = {}
    for i, key in enumerate(CORNERS):
        _, _, _, max_loc = cv2.minMaxLoc(heat[i])
        verts[key] = (float(max_loc[0]), float(max_loc[1]))
    return verts


class JalilianDataset(Dataset):
    def __init__(self, root: Path, split: str, size: int = 384, augment: bool = False) -> None:
        self.root = Path(root)
        self.split = split
        self.size = size
        self.augment = augment
        self.img_dir, self.mask_dir = db_dirs(self.root, split)
        self.ids = list_ids(self.root, split)
        if not self.ids:
            raise FileNotFoundError(f"이미지/마스크 없음: {self.img_dir} + {self.mask_dir}")
        self.expert = load_expert_vertices(self.root, split)

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int) -> dict:
        stem = self.ids[idx]
        image = preprocess_for_model(load_gray(self.img_dir / f"{stem}.png"))
        mask = load_mask(self.mask_dir / f"{stem}.png")
        verts = self.expert.get(stem) or vertices_from_mask_extrema(mask)
        if verts is None:
            h, w = image.shape[:2]
            verts = {"N": (w / 2, 0.0), "E": (float(w), h / 2), "S": (w / 2, float(h)), "W": (0.0, h / 2)}

        if self.augment and np.random.rand() < 0.65:
            image, mask, verts = _crop_to_indent(image, mask, verts)

        h0, w0 = image.shape[:2]
        image = cv2.resize(image, (self.size, self.size), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, (self.size, self.size), interpolation=cv2.INTER_NEAREST)
        verts = _scale_verts(verts, self.size / w0, self.size / h0)
        if self.augment:
            image, mask, verts = _augment(image, mask, verts)

        _, _, dmean = diagonals_px(verts)
        heat = make_heatmaps(self.size, self.size, verts, max(2.5, 0.035 * dmean))
        return {
            "image": torch.from_numpy(image.astype(np.float32) / 255.0).unsqueeze(0),
            "mask": torch.from_numpy((mask > 127).astype(np.float32)).unsqueeze(0),
            "heat": torch.from_numpy(heat),
            "verts": torch.tensor([[verts[k][0], verts[k][1]] for k in CORNERS], dtype=torch.float32),
            "id": stem,
        }


def _scale_verts(verts: dict[str, Vertex], sx: float, sy: float) -> dict[str, Vertex]:
    return {k: (p[0] * sx, p[1] * sy) for k, p in verts.items()}


def _crop_to_indent(
    image: np.ndarray,
    mask: np.ndarray,
    verts: dict[str, Vertex],
) -> tuple[np.ndarray, np.ndarray, dict[str, Vertex]]:
    h, w = image.shape[:2]
    ys, xs = np.where(mask > 127)
    if len(xs) < 8:
        return image, mask, verts
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    bw, bh = max(x1 - x0 + 1, 8), max(y1 - y0 + 1, 8)
    crop_s = max(int(max(bw, bh) * float(np.random.uniform(1.4, 3.2))), 64)
    cx = 0.5 * (x0 + x1) + float(np.random.uniform(-0.12, 0.12)) * bw
    cy = 0.5 * (y0 + y1) + float(np.random.uniform(-0.12, 0.12)) * bh
    x_a = int(round(cx - crop_s / 2))
    y_a = int(round(cy - crop_s / 2))
    x_b, y_b = x_a + crop_s, y_a + crop_s
    x_a0, y_a0 = x_a, y_a
    pad_l, pad_t = max(0, -x_a), max(0, -y_a)
    pad_r, pad_b = max(0, x_b - w), max(0, y_b - h)
    if pad_l or pad_t or pad_r or pad_b:
        image = cv2.copyMakeBorder(image, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_REFLECT_101)
        mask = cv2.copyMakeBorder(mask, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_CONSTANT, value=0)
        x_a += pad_l
        y_a += pad_t
        x_b += pad_l
        y_b += pad_t
    image = image[y_a:y_b, x_a:x_b]
    mask = mask[y_a:y_b, x_a:x_b]
    verts = {k: (p[0] - x_a0, p[1] - y_a0) for k, p in verts.items()}
    return image, mask, verts


def _augment(
    image: np.ndarray,
    mask: np.ndarray,
    verts: dict[str, Vertex],
) -> tuple[np.ndarray, np.ndarray, dict[str, Vertex]]:
    h, w = image.shape[:2]
    if np.random.rand() < 0.5:
        image, mask = cv2.flip(image, 1), cv2.flip(mask, 1)
        verts = {k: (w - 1 - p[0], p[1]) for k, p in verts.items()}
        verts["E"], verts["W"] = verts["W"], verts["E"]
    if np.random.rand() < 0.5:
        h, w = image.shape[:2]
        image, mask = cv2.flip(image, 0), cv2.flip(mask, 0)
        verts = {k: (p[0], h - 1 - p[1]) for k, p in verts.items()}
        verts["N"], verts["S"] = verts["S"], verts["N"]
    k = int(np.random.randint(0, 4))
    if k:
        image, mask, verts = _rot90(image, mask, verts, k)
    if np.random.rand() < 0.5:
        h, w = image.shape[:2]
        angle = float(np.random.uniform(-12, 12))
        mat = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
        image = cv2.warpAffine(image, mat, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        mask = cv2.warpAffine(mask, mat, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT)
        verts = _warp_verts(verts, mat)
    alpha = 0.7 + 0.6 * np.random.rand()
    beta = np.random.uniform(-22, 22)
    image = np.clip(image.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
    if np.random.rand() < 0.25:
        image = cv2.GaussianBlur(image, (3, 3), 0)
    if np.random.rand() < 0.25:
        noise = np.random.normal(0.0, 7.0, image.shape).astype(np.float32)
        image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return image, mask, verts


def _rot90(
    image: np.ndarray,
    mask: np.ndarray,
    verts: dict[str, Vertex],
    k: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Vertex]]:
    k = k % 4
    h, w = image.shape[:2]
    image = np.rot90(image, k).copy()
    mask = np.rot90(mask, k).copy()
    perm = {"N": "W", "W": "S", "S": "E", "E": "N"}
    cur = verts
    hh, ww = h, w
    for _ in range(k):
        nxt: dict[str, Vertex] = {}
        for src, dst in perm.items():
            x, y = cur[src]
            nxt[dst] = (float(y), float(ww - 1 - x))
        cur = nxt
        hh, ww = ww, hh
    return image, mask, cur


def _warp_verts(verts: dict[str, Vertex], mat: np.ndarray) -> dict[str, Vertex]:
    out: dict[str, Vertex] = {}
    for k, (x, y) in verts.items():
        nx = mat[0, 0] * x + mat[0, 1] * y + mat[0, 2]
        ny = mat[1, 0] * x + mat[1, 1] * y + mat[1, 2]
        out[k] = (float(nx), float(ny))
    return out


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
    """4단 U-Net. 채널 0=마스크, 1..4=N/E/S/W 히트맵."""

    def __init__(self, base: int = 32, out_ch: int = 5) -> None:
        super().__init__()
        self.enc1 = ConvBlock(1, base)
        self.enc2 = ConvBlock(base, base * 2)
        self.enc3 = ConvBlock(base * 2, base * 4)
        self.enc4 = ConvBlock(base * 4, base * 8)
        self.bot = ConvBlock(base * 8, base * 16)
        self.pool = nn.MaxPool2d(2)
        self.drop = nn.Dropout2d(0.10)
        self.up4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(base * 16, base * 8, 1, bias=False),
        )
        self.dec4 = ConvBlock(base * 16, base * 8)
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(base * 8, base * 4, 1, bias=False),
        )
        self.dec3 = ConvBlock(base * 8, base * 4)
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(base * 4, base * 2, 1, bias=False),
        )
        self.dec2 = ConvBlock(base * 4, base * 2)
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(base * 2, base, 1, bias=False),
        )
        self.dec1 = ConvBlock(base * 2, base)
        self.out = nn.Conv2d(base, out_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))
        b = self.drop(self.bot(self.pool(e4)))
        d4 = self.dec4(torch.cat([self.up4(b), e4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.out(d1)


def _boundary_weight(target: torch.Tensor, k: int = 5) -> torch.Tensor:
    dil = nn.functional.max_pool2d(target, k, stride=1, padding=k // 2)
    ero = 1.0 - nn.functional.max_pool2d(1.0 - target, k, stride=1, padding=k // 2)
    return 1.0 + 5.0 * (dil - ero).clamp(0, 1)


def mask_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    weight = _boundary_weight(target)
    bce = nn.functional.binary_cross_entropy_with_logits(logits, target, weight=weight)
    pred = torch.sigmoid(logits)
    num = 2 * (pred * target).sum(dim=(1, 2, 3))
    den = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + eps
    dice = 1 - (num / den).mean()
    tp = (pred * target).sum(dim=(1, 2, 3))
    fp = (pred * (1 - target)).sum(dim=(1, 2, 3))
    fn = ((1 - pred) * target).sum(dim=(1, 2, 3))
    tversky = 1 - (tp / (tp + 0.3 * fp + 0.7 * fn + eps)).mean()
    return bce + dice + 0.5 * tversky


def multitask_loss(logits: torch.Tensor, mask: torch.Tensor, heat: torch.Tensor) -> torch.Tensor:
    """마스크 + 꼭짓점 히트맵. 히트맵 쪽 가중치를 더 줌."""
    return mask_loss(logits[:, 0:1], mask) + 6.0 * nn.functional.mse_loss(torch.sigmoid(logits[:, 1:5]), heat)


def bce_dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """이전 학습 루프 호환. 5채널이면 마스크만 사용."""
    return mask_loss(logits[:, 0:1], target, eps=eps)


@torch.no_grad()
def dice_score(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> float:
    pred = (torch.sigmoid(logits[:, 0:1]) > 0.5).float()
    num = 2 * (pred * target).sum(dim=(1, 2, 3))
    den = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + eps
    return float((num / den).mean().item())


@torch.no_grad()
def mean_diag_error(logits: torch.Tensor, verts: torch.Tensor) -> float | None:
    """히트맵 꼭짓점 vs 전문가 꼭짓점의 평균 대각선 상대오차."""
    heat = torch.sigmoid(logits[:, 1:5]).cpu().numpy()
    gt = verts.detach().cpu().numpy()
    errs: list[float] = []
    for i in range(heat.shape[0]):
        pred = peaks_from_heatmaps(heat[i])
        gv = {k: (float(gt[i, j, 0]), float(gt[i, j, 1])) for j, k in enumerate(CORNERS)}
        _, _, dp = diagonals_px(pred)
        _, _, dg = diagonals_px(gv)
        if dg > 1.0:
            errs.append(abs(dp - dg) / dg)
    if not errs:
        return None
    return float(sum(errs) / len(errs))


@torch.no_grad()
def _forward_full(model: nn.Module, gray: np.ndarray, size: int, device: torch.device) -> np.ndarray:
    inp = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
    tensor = torch.from_numpy(inp.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0).to(device)
    logits = model(tensor)[0].detach().cpu().numpy()
    return logits


def predict_indent(
    model: nn.Module,
    gray: np.ndarray,
    size: int,
    device: torch.device,
) -> tuple[np.ndarray, dict[str, Vertex] | None]:
    """전체 이미지 → 크롭 재추론. 마스크와 N/E/S/W 히트맵 꼭짓점."""
    h, w = gray.shape[:2]
    work = preprocess_for_model(gray)
    logits = _forward_full(model, work, size, device)
    mask_s = (1.0 / (1.0 + np.exp(-logits[0])) > 0.5).astype(np.uint8) * 255
    mask_s = cv2.resize(mask_s, (w, h), interpolation=cv2.INTER_NEAREST)
    ys, xs = np.where(mask_s > 127)
    if len(xs) >= 8:
        x0, x1 = int(xs.min()), int(xs.max())
        y0, y1 = int(ys.min()), int(ys.max())
        bw, bh = max(x1 - x0 + 1, 8), max(y1 - y0 + 1, 8)
        if max(bw / w, bh / h) < 0.85:
            pad = int(0.4 * max(bw, bh))
            xa, ya = max(0, x0 - pad), max(0, y0 - pad)
            xb, yb = min(w, x1 + pad + 1), min(h, y1 + pad + 1)
            crop = work[ya:yb, xa:xb]
            logits = _forward_full(model, crop, size, device)
            ch, cw = crop.shape[:2]
            mask = (1.0 / (1.0 + np.exp(-logits[0])) > 0.5).astype(np.uint8) * 255
            mask = cv2.resize(mask, (cw, ch), interpolation=cv2.INTER_NEAREST)
            full = np.zeros((h, w), dtype=np.uint8)
            full[ya:yb, xa:xb] = mask
            heat = 1.0 / (1.0 + np.exp(-logits[1:5]))
            heat_r = np.stack([cv2.resize(heat[i], (cw, ch), interpolation=cv2.INTER_LINEAR) for i in range(4)])
            verts = peaks_from_heatmaps(heat_r)
            verts = {k: (p[0] + xa, p[1] + ya) for k, p in verts.items()}
            return full, verts
    heat = 1.0 / (1.0 + np.exp(-logits[1:5]))
    heat_r = np.stack([cv2.resize(heat[i], (w, h), interpolation=cv2.INTER_LINEAR) for i in range(4)])
    return mask_s, peaks_from_heatmaps(heat_r)


def vertices_from_mask_extrema(mask: np.ndarray) -> dict[str, Vertex] | None:
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


def vertices_from_mask(mask: np.ndarray) -> dict[str, Vertex] | None:
    """마스크 폴백용. 학습된 히트맵이 있으면 predict_indent를 쓴다."""
    return vertices_from_mask_extrema(mask)


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
        cv2.putText(vis, "no contour", (24, 48), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 255), 2)
        return vis
    pts = [tuple(map(int, verts[k])) for k in CORNERS]
    cv2.line(vis, pts[0], pts[2], color, 2, cv2.LINE_AA)
    cv2.line(vis, pts[1], pts[3], color, 2, cv2.LINE_AA)
    for key, pt in zip(CORNERS, pts):
        cv2.circle(vis, pt, 8, color, -1, cv2.LINE_AA)
        cv2.putText(vis, key, (pt[0] + 10, pt[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    d1, d2, dmean = diagonals_px(verts)
    text = f"d1={d1:.1f}px  d2={d2:.1f}px  mean={dmean:.1f}px"
    cv2.putText(vis, text, (20, vis.shape[0] - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return vis


def _optimizer_step(loss, opt, model, kind: str, xm, scaler) -> None:
    if kind == "cuda":
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        return
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    if kind == "tpu":
        xm.optimizer_step(opt)
    else:
        opt.step()


def _save_ckpt(path: Path, model: nn.Module, extra: dict, kind: str, xm) -> None:
    blob = {
        "model": {k: v.detach().to("cpu") for k, v in model.state_dict().items()},
        **extra,
    }
    if kind == "tpu":
        xm.save(blob, str(path))
    else:
        torch.save(blob, path)


def main() -> None:
    data = find_data_root()
    out = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path("weights")
    out.mkdir(parents=True, exist_ok=True)
    device, kind, xm = pick_accelerator()
    if kind == "cpu" and Path("/kaggle").exists():
        print(
            "경고: CPU로 학습합니다. Kaggle 오른쪽 Accelerator가 TPU(또는 GPU)인지 확인하세요.",
            flush=True,
        )
    batch = 8 if kind in ("tpu", "cuda") else 4
    use_amp = kind == "cuda"
    print("data =", data)
    extra = ""
    if kind == "cuda":
        extra = torch.cuda.get_device_name(0)
    elif kind == "tpu":
        extra = os.environ.get("TPU_NAME") or os.environ.get("PJRT_DEVICE") or "xla"
    print("device =", device, "kind =", kind, extra)
    train_ds = JalilianDataset(data, "I", size=SIZE, augment=True)
    val_ds = JalilianDataset(data, "II", size=SIZE, augment=False)
    print(f"expert verts train={len(train_ds.expert)}/{len(train_ds)}  val={len(val_ds.expert)}/{len(val_ds)}")
    nw = 2 if kind == "cuda" else 0
    train_loader = DataLoader(train_ds, batch_size=batch, shuffle=True, num_workers=nw)
    val_loader = DataLoader(val_ds, batch_size=batch, shuffle=False, num_workers=nw)
    model = UNet(base=32, out_ch=5).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    warmup = torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.1, total_iters=3)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(EPOCHS - 3, 1), eta_min=1e-5)
    sched = torch.optim.lr_scheduler.SequentialLR(opt, [warmup, cosine], milestones=[3])
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    print(f"train={len(train_ds)} val={len(val_ds)} batch={batch} epochs={EPOCHS} size={SIZE} lr={LR}")
    best_dice = -1.0
    best_diag = 1e9
    for epoch in range(1, EPOCHS + 1):
        model.train()
        running = 0.0
        train_dices = []
        for batch_data in train_loader:
            x = batch_data["image"].to(device)
            y = batch_data["mask"].to(device)
            heat = batch_data["heat"].to(device)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(x)
                loss = multitask_loss(logits, y, heat)
            _optimizer_step(loss, opt, model, kind, xm, scaler)
            running += float(loss.item()) * x.size(0)
            train_dices.append(dice_score(logits.detach(), y))
        train_loss = running / max(len(train_ds), 1)
        train_dice = sum(train_dices) / max(len(train_dices), 1)
        sched.step()
        model.eval()
        dices, diag_errs = [], []
        with torch.no_grad():
            for batch_data in val_loader:
                x = batch_data["image"].to(device)
                y = batch_data["mask"].to(device)
                with torch.cuda.amp.autocast(enabled=use_amp):
                    logits = model(x)
                dices.append(dice_score(logits, y))
                err = mean_diag_error(logits, batch_data["verts"])
                if err is not None:
                    diag_errs.append(err)
        val_dice = sum(dices) / max(len(dices), 1)
        val_diag = sum(diag_errs) / max(len(diag_errs), 1) if diag_errs else None
        mark = ""
        should_save = False
        if val_diag is not None and val_dice >= 0.85 and val_diag < best_diag:
            best_diag = val_diag
            should_save = True
        elif best_diag >= 1e8 and val_dice > best_dice:
            should_save = True
        if val_dice > best_dice:
            best_dice = val_dice
        if should_save:
            _save_ckpt(
                out / "best.pt",
                model,
                {
                    "size": SIZE,
                    "epoch": epoch,
                    "val_dice": val_dice,
                    "val_diag": val_diag,
                    "arch": "unet4_heat5",
                },
                kind,
                xm,
            )
            mark = "  <- saved"
        lr_now = opt.param_groups[0]["lr"]
        diag_txt = f"{100 * val_diag:.2f}%" if val_diag is not None else "n/a"
        print(
            f"epoch {epoch:03d}/{EPOCHS}  loss={train_loss:.4f}  train_dice={train_dice:.4f}  "
            f"val_dice={val_dice:.4f}  val_diag={diag_txt}  lr={lr_now:.2e}{mark}",
            flush=True,
        )
    print("best val_dice =", round(best_dice, 4))
    print("best val_diag =", f"{100 * best_diag:.2f}%" if best_diag < 1e8 else "n/a")
    print("weights =", out / "best.pt")


main()
