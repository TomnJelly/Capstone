#!/usr/bin/env python3
"""v2: 1채널 마스크 U-Net. DB마다 70/30. 꼭짓점은 마스크에서 뽑는다."""

from __future__ import annotations

import json
import os
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

Vertex = tuple[float, float]
CORNERS = ("N", "E", "S", "W")
ARCH = "unet1_mask384_v2"
DEFAULT_SIZE = 384
DEFAULT_BASE = 48


def is_vicker_root(path: Path) -> bool:
    return (path / "DB-I").is_dir() and (path / "exact-masks-I").is_dir()


def find_data_root(explicit: Path | None = None) -> Path:
    def check(path: Path) -> Path | None:
        try:
            path = path.expanduser().resolve()
        except OSError:
            return None
        if is_vicker_root(path):
            return path
        nested = path / "Vicker_Database"
        if is_vicker_root(nested):
            return nested
        return None

    if explicit is not None:
        found = check(Path(explicit))
        if found is not None:
            return found
        raise FileNotFoundError(
            f"{explicit} 에서 데이터를 찾지 못했습니다.\n"
            "그 폴더 안에 DB-I 와 exact-masks-I 가 있어야 합니다."
        )

    kaggle = Path("/kaggle/input")
    if kaggle.exists():
        for mask_dir in kaggle.rglob("exact-masks-I"):
            found = check(mask_dir.parent)
            if found is not None:
                return found
        for child in sorted(kaggle.iterdir()):
            found = check(child)
            if found is not None:
                return found
        names = [p.name for p in kaggle.iterdir()]
        raise FileNotFoundError(
            f"/kaggle/input 아래에 DB-I + exact-masks-I 가 없습니다. 지금 있는 것: {names}\n"
            "Notebook에서 Vicker_Database 를 Add Input 하세요."
        )

    env = os.environ.get("VICKER_DATABASE")
    try:
        here = Path(__file__).resolve().parent
    except NameError:
        here = Path.cwd()
    candidates = [
        Path(env) if env else None,
        here / "Vicker_Database",
        here.parent / "Vicker_Database",
        Path.cwd() / "Vicker_Database",
        Path.home() / "Downloads" / "Vicker_Database",
        Path.home() / "Download" / "Vicker_Database",
        Path.home() / "Documents" / "Vicker_Database",
        Path(r"C:\Users\seoyu\Downloads\Vicker_Database"),
    ]
    seen: set[Path] = set()
    for raw in candidates:
        if raw is None:
            continue
        found = check(raw)
        if found is None or found in seen:
            continue
        seen.add(found)
        return found
    raise FileNotFoundError("Vicker_Database 를 찾지 못했습니다. --data 로 경로를 넣으세요.")


def db_dirs(root: Path, split: str) -> tuple[Path, Path]:
    tag = split.upper().replace("DB-", "")
    return root / f"DB-{tag}", root / f"exact-masks-{tag}"


def list_ids(root: Path, split: str) -> list[str]:
    img_dir, mask_dir = db_dirs(root, split)
    return [p.stem for p in sorted(img_dir.glob("*.png")) if (mask_dir / p.name).exists()]


def split_70_30(ids: list[str], seed: int = 42) -> tuple[list[str], list[str]]:
    rng = np.random.RandomState(seed)
    order = list(ids)
    rng.shuffle(order)
    n_train = int(round(len(order) * 0.7))
    n_train = min(max(n_train, 1), len(order) - 1) if len(order) > 1 else len(order)
    return order[:n_train], order[n_train:]


def save_splits(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_gray(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    return img


def load_mask(path: Path) -> np.ndarray:
    mask = load_gray(path)
    return np.where(mask > 127, 255, 0).astype(np.uint8)


def enhance_gray(gray: np.ndarray) -> np.ndarray:
    """저대비 현미경 사진에서 압흔 경계를 살린다. 학습/추론 동일하게 쓴다."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def clean_mask(mask: np.ndarray) -> np.ndarray:
    """여러 덩어리 중 가운데에 가까운 큰 압흔만 남긴다. v1 1003/1032 대응."""
    binary = (mask > 127).astype(np.uint8)
    n, labels, stats, cents = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if n <= 1:
        return np.zeros_like(mask)
    h, w = mask.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    areas = stats[1:, cv2.CC_STAT_AREA]
    max_area = float(areas.max())
    best_i = 1 + int(np.argmax(areas))
    best_score = -1.0
    for i in range(1, n):
        area = float(stats[i, cv2.CC_STAT_AREA])
        if area < 20 or area < 0.12 * max_area:
            continue
        dist = float(np.hypot(cents[i][0] - cx, cents[i][1] - cy))
        score = area / (1.0 + dist / 80.0)
        if score > best_score:
            best_score = score
            best_i = i
    return np.where(labels == best_i, 255, 0).astype(np.uint8)


class JalilianSubset(Dataset):
    def __init__(
        self,
        root: Path,
        split: str,
        ids: list[str],
        size: int = DEFAULT_SIZE,
        augment: bool = False,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.size = size
        self.augment = augment
        self.img_dir, self.mask_dir = db_dirs(self.root, split)
        self.ids = list(ids)
        if not self.ids:
            raise FileNotFoundError(f"이미지 없음: {self.img_dir}")

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int) -> dict:
        stem = self.ids[idx]
        image = enhance_gray(load_gray(self.img_dir / f"{stem}.png"))
        mask = load_mask(self.mask_dir / f"{stem}.png")
        image = cv2.resize(image, (self.size, self.size), interpolation=cv2.INTER_AREA)
        mask = cv2.resize(mask, (self.size, self.size), interpolation=cv2.INTER_NEAREST)
        if self.augment:
            image, mask = _augment(image, mask)
        return {
            "image": torch.from_numpy(image.astype(np.float32) / 255.0).unsqueeze(0),
            "mask": torch.from_numpy((mask > 127).astype(np.float32)).unsqueeze(0),
            "id": stem,
        }


def _augment(image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if np.random.rand() < 0.5:
        image, mask = cv2.flip(image, 1), cv2.flip(mask, 1)
    if np.random.rand() < 0.5:
        image, mask = cv2.flip(image, 0), cv2.flip(mask, 0)
    k = int(np.random.randint(0, 4))
    if k:
        image, mask = np.rot90(image, k).copy(), np.rot90(mask, k).copy()
    if np.random.rand() < 0.5:
        angle = float(np.random.uniform(-12, 12))
        h, w = image.shape[:2]
        mat = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
        image = cv2.warpAffine(image, mat, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        mask = cv2.warpAffine(mask, mat, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    if np.random.rand() < 0.5:
        scale = float(np.random.uniform(0.88, 1.14))
        image, mask = _zoom(image, mask, scale)
    alpha = 0.7 + 0.6 * np.random.rand()
    beta = np.random.randint(-22, 23)
    image = np.clip(image.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
    if np.random.rand() < 0.25:
        image = cv2.GaussianBlur(image, (3, 3), 0.6)
    if np.random.rand() < 0.35:
        noise = np.random.normal(0, 6.0, image.shape).astype(np.float32)
        image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return image, mask


def _zoom(image: np.ndarray, mask: np.ndarray, scale: float) -> tuple[np.ndarray, np.ndarray]:
    h, w = image.shape[:2]
    nh, nw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
    interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    img2 = cv2.resize(image, (nw, nh), interpolation=interp)
    m2 = cv2.resize(mask, (nw, nh), interpolation=cv2.INTER_NEAREST)
    if scale < 1.0:
        y0 = (h - nh) // 2
        x0 = (w - nw) // 2
        y1 = h - nh - y0
        x1 = w - nw - x0
        img2 = cv2.copyMakeBorder(img2, y0, y1, x0, x1, cv2.BORDER_REFLECT_101)[:h, :w]
        m2 = cv2.copyMakeBorder(m2, y0, y1, x0, x1, cv2.BORDER_CONSTANT, value=0)[:h, :w]
        return img2, m2
    y0 = max(0, (nh - h) // 2)
    x0 = max(0, (nw - w) // 2)
    return img2[y0 : y0 + h, x0 : x0 + w], m2[y0 : y0 + h, x0 : x0 + w]


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
    """1채널 마스크 U-Net. v1보다 채널을 조금 키우고 bottleneck dropout."""

    def __init__(self, base: int = DEFAULT_BASE) -> None:
        super().__init__()
        self.base = base
        self.enc1 = ConvBlock(1, base)
        self.enc2 = ConvBlock(base, base * 2)
        self.enc3 = ConvBlock(base * 2, base * 4)
        self.enc4 = ConvBlock(base * 4, base * 8)
        self.pool = nn.MaxPool2d(2)
        self.drop = nn.Dropout2d(0.10)
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
        e4 = self.drop(self.enc4(self.pool(e3)))
        d3 = self.dec3(torch.cat([self.up3(e4), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.out(d1)


def _edge_band(target: torch.Tensor) -> torch.Tensor:
    dil = nn.functional.max_pool2d(target, 3, stride=1, padding=1)
    ero = 1.0 - nn.functional.max_pool2d(1.0 - target, 3, stride=1, padding=1)
    return (dil - ero).clamp(0.0, 1.0)


def bce_dice_boundary_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    bce = nn.functional.binary_cross_entropy_with_logits(logits, target)
    pred = torch.sigmoid(logits)
    num = 2 * (pred * target).sum(dim=(1, 2, 3))
    den = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + eps
    dice = 1 - (num / den).mean()
    edge = _edge_band(target)
    per = nn.functional.binary_cross_entropy_with_logits(logits, target, reduction="none")
    edge_bce = (per * edge).sum() / edge.sum().clamp_min(1.0)
    return bce + dice + 0.5 * edge_bce


@torch.no_grad()
def dice_score(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> float:
    pred = (torch.sigmoid(logits.float()) > 0.5).float()
    num = 2 * (pred * target).sum(dim=(1, 2, 3))
    den = pred.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) + eps
    return float((num / den).mean().item())


def _forward_prob(model: nn.Module, gray_resized: np.ndarray, device: torch.device) -> np.ndarray:
    tensor = torch.from_numpy(gray_resized.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0).to(device)
    return torch.sigmoid(model(tensor).float())[0, 0].cpu().numpy()


@torch.no_grad()
def predict_mask(
    model: nn.Module,
    gray: np.ndarray,
    size: int,
    device: torch.device,
    tta: bool = True,
) -> np.ndarray:
    h, w = gray.shape[:2]
    inp = cv2.resize(enhance_gray(gray), (size, size), interpolation=cv2.INTER_AREA)
    model.eval()
    acc = _forward_prob(model, inp, device)
    if tta:
        acc = acc.copy()
        n = 1.0
        for code in (1, 0):
            flipped = cv2.flip(inp, code)
            pred = _forward_prob(model, flipped, device)
            acc += cv2.flip(pred, code)
            n += 1.0
        both = cv2.flip(inp, -1)
        pred = _forward_prob(model, both, device)
        acc += cv2.flip(pred, -1)
        n += 1.0
        acc /= n
    pred = (acc > 0.5).astype(np.uint8) * 255
    pred = cv2.resize(pred, (w, h), interpolation=cv2.INTER_NEAREST)
    return clean_mask(pred)


def vertices_from_mask(mask: np.ndarray) -> dict[str, Vertex] | None:
    cleaned = clean_mask(mask)
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea)
    if cv2.contourArea(cnt) < 20:
        return None
    hull = cv2.convexHull(cnt).reshape(-1, 2)
    pts = hull if len(hull) >= 4 else cnt.reshape(-1, 2)
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
