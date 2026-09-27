#!/usr/bin/env python3
"""
비커스 학습 입출력을 단계별로 보는 HTML 뷰어.

학습 전: 원본 → GT 마스크 → 오버레이 → 정답 꼭짓점
학습 후: 같은 단계 + 예측 마스크 → 예측 꼭짓점
왼쪽 목록에 장별 대각선 오차, 헤더에 전체 요약을 붙인다.
"""

from __future__ import annotations

import argparse
import http.server
import json
import socketserver
import sys
import threading
import time
import webbrowser
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from common import (  # noqa: E402
    UNet,
    db_dirs,
    diagonals_px,
    draw_geometry,
    find_data_root,
    list_ids,
    load_gray,
    load_mask,
    overlay_mask,
    predict_mask,
    split_70_30,
    vertices_from_mask,
)

STEPS_GT = [
    ("01_input.png", "1. 입력 이미지", "시험기 현미경 흑백 사진. 모델의 입력입니다."),
    ("02_gt_mask.png", "2. 정답 마스크", "사람이 칠한 압흔 영역. 학습 타깃입니다."),
    ("03_gt_overlay.png", "3. 입력 + 정답", "초록이 정답 압흔입니다."),
    ("04_gt_geom.png", "4. 정답 기하", "정답 마스크에서 뽑은 N/E/S/W와 대각선."),
]

STEPS_PRED = [
    ("05_pred_mask.png", "5. 모델 예측 마스크", "U-Net이 예측한 압흔 영역."),
    ("06_pred_geom.png", "6. 예측 기하", "예측 마스크에서 뽑은 꼭짓점. 초록=정답, 주황=예측."),
]

GOOD_MAX = 0.03
OK_MAX = 0.08


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Vickers step-by-step preview")
    p.add_argument("--data", type=Path, default=None, help="Vicker_Database 폴더. 생략하면 자동으로 찾습니다.")
    p.add_argument("--out", type=Path, default=ROOT / "preview")
    p.add_argument("--ckpt", type=Path, default=None, help="학습된 pt. 생략하면 v2/weights/best_I.pt")
    p.add_argument(
        "--samples",
        type=int,
        default=0,
        help="몇 장까지 볼지. 0이면 --subset 전부(검증 30%면 그 전부).",
    )
    p.add_argument("--split", default="I", help="I 또는 II")
    p.add_argument(
        "--subset",
        choices=("val", "train", "all"),
        default="val",
        help="기본은 학습에 안 넣은 검증 30%.",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-open", action="store_true")
    p.add_argument("--port", type=int, default=8766)
    return p.parse_args()


def pick_ids(root: Path, split: str, n: int, subset: str, seed: int) -> tuple[list[str], int, int]:
    ids = list_ids(root, split)
    if not ids:
        raise FileNotFoundError(f"샘플 없음: DB-{split}")
    train_ids, val_ids = split_70_30(ids, seed=seed)
    if subset == "val":
        chosen = val_ids
    elif subset == "train":
        chosen = train_ids
    else:
        chosen = ids
    chosen = sorted(chosen)
    if n > 0:
        chosen = chosen[:n]
    return chosen, len(train_ids), len(val_ids)


def load_model(ckpt_path: Path, device: torch.device) -> tuple[nn.Module, int, str, dict]:
    blob = torch.load(ckpt_path, map_location=device, weights_only=False)
    size = int(blob.get("size", 384))
    out_ch = int(blob["model"]["out.weight"].shape[0])
    if out_ch != 1:
        raise SystemExit(
            f"{ckpt_path} 는 1채널 마스크 모델이 아닙니다 (out_ch={out_ch})."
        )
    base = int(blob.get("base", 48))
    model: nn.Module = UNet(base=base)
    model = model.to(device)
    model.load_state_dict(blob["model"])
    model.eval()
    meta = {
        "epoch": blob.get("epoch"),
        "val_dice": blob.get("val_dice"),
        "val_diag": blob.get("val_diag"),
        "arch": blob.get("arch") or "unet1_mask384_v2",
        "size": size,
        "kind": "mask1",
        "db": blob.get("db"),
    }
    return model, size, "mask1", meta


def mask_dice(pred: np.ndarray, gt: np.ndarray) -> float:
    p = pred > 127
    g = gt > 127
    inter = np.logical_and(p, g).sum()
    den = p.sum() + g.sum()
    if den == 0:
        return 0.0
    return float(2.0 * inter / den)


def grade_of(diag: float | None) -> str:
    if diag is None:
        return "na"
    if diag < GOOD_MAX:
        return "good"
    if diag < OK_MAX:
        return "ok"
    return "bad"


def sample_metrics(
    gt: np.ndarray,
    pred: np.ndarray | None,
    gt_verts: dict | None,
    pred_verts: dict | None,
) -> dict:
    dice = mask_dice(pred, gt) if pred is not None else None
    diag = None
    gt_mean = None
    pred_mean = None
    if gt_verts is not None and pred_verts is not None:
        _, _, dg = diagonals_px(gt_verts)
        _, _, dp = diagonals_px(pred_verts)
        gt_mean = dg
        pred_mean = dp
        if dg > 1.0:
            diag = abs(dp - dg) / dg
    return {
        "dice": dice,
        "diag": diag,
        "gt_mean": gt_mean,
        "pred_mean": pred_mean,
        "grade": grade_of(diag),
    }


def save_steps(
    stem: str,
    gray: np.ndarray,
    gt: np.ndarray,
    dest: Path,
    pred: np.ndarray | None,
    gt_verts: dict | None = None,
    pred_verts: dict | None = None,
    metrics: dict | None = None,
) -> dict:
    dest.mkdir(parents=True, exist_ok=True)
    if gt_verts is None:
        gt_verts = vertices_from_mask(gt)
    cv2.imwrite(str(dest / "01_input.png"), gray)
    cv2.imwrite(str(dest / "02_gt_mask.png"), gt)
    cv2.imwrite(str(dest / "03_gt_overlay.png"), overlay_mask(gray, gt, (46, 204, 113)))
    cv2.imwrite(str(dest / "04_gt_geom.png"), draw_geometry(gray, gt_verts, (46, 180, 90)))

    has_pred = pred is not None
    if has_pred and pred is not None:
        if pred_verts is None:
            pred_verts = vertices_from_mask(pred)
        cv2.imwrite(str(dest / "05_pred_mask.png"), overlay_mask(gray, pred, (231, 76, 60)))
        pred_vis = draw_geometry(gray, pred_verts, (0, 70, 255))
        if gt_verts is not None:
            for pt in gt_verts.values():
                cv2.circle(pred_vis, (int(pt[0]), int(pt[1])), 6, (46, 204, 113), 2)
        cv2.imwrite(str(dest / "06_pred_geom.png"), pred_vis)

    return {
        "id": stem,
        "has_pred": has_pred,
        "gt_verts": gt_verts,
        "pred_verts": pred_verts,
    }


def summarize(items: list[dict], meta: dict | None) -> dict:
    diags = [float(it["diag"]) for it in items if it.get("diag") is not None]
    dices = [float(it["dice"]) for it in items if it.get("dice") is not None]
    n_good = sum(1 for it in items if it.get("grade") == "good")
    n_ok = sum(1 for it in items if it.get("grade") == "ok")
    n_bad = sum(1 for it in items if it.get("grade") == "bad")
    n_na = sum(1 for it in items if it.get("grade") == "na")
    mean_diag = sum(diags) / len(diags) if diags else None
    median_diag = float(np.median(diags)) if diags else None
    mean_dice = sum(dices) / len(dices) if dices else None
    edges = [0.0, 0.03, 0.08, 0.15, 1e9]
    labels = ["<3%", "3–8%", "8–15%", ">15%"]
    hist = []
    for i, label in enumerate(labels):
        lo, hi = edges[i], edges[i + 1]
        hist.append({"label": label, "n": sum(1 for d in diags if lo <= d < hi)})
    return {
        "n": len(items),
        "n_scored": len(diags),
        "mean_diag": mean_diag,
        "median_diag": median_diag,
        "mean_dice": mean_dice,
        "n_good": n_good,
        "n_ok": n_ok,
        "n_bad": n_bad,
        "n_na": n_na,
        "hist": hist,
        "ckpt": meta or {},
        "subset": (meta or {}).get("_subset"),
        "train_n": (meta or {}).get("_train_n"),
        "val_n": (meta or {}).get("_val_n"),
    }


def write_html(out: Path, items: list[dict], has_model: bool, summary: dict, bust: str) -> None:
    steps = STEPS_GT + (STEPS_PRED if has_model else [])
    payload = json.dumps(
        {
            "items": items,
            "steps": [[s[0], s[1]] for s in steps],
            "has_model": has_model,
            "summary": summary,
            "bust": bust,
        },
        ensure_ascii=False,
    )
    html = f"""<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>비커스 학습 입출력 미리보기</title>
  <style>
    :root {{ --bg:#0e1117; --card:#171b22; --line:#2a3140; --text:#e8edf5; --muted:#9aa6b8; --accent:#3b82f6;
      --good:#34d399; --ok:#fbbf24; --bad:#f87171; }}
    * {{ box-sizing: border-box; }}
    body {{ margin:0; font-family:"Segoe UI",sans-serif; background:var(--bg); color:var(--text); }}
    header {{ padding:14px 20px 12px; border-bottom:1px solid var(--line); }}
    h1 {{ margin:0 0 6px; font-size:24px; }}
    .sub {{ color:var(--muted); font-size:16px; margin-bottom:12px; }}
    .stats {{ display:flex; flex-wrap:wrap; gap:12px; align-items:stretch; }}
    .stat {{ background:var(--card); border:1px solid var(--line); border-radius:10px; padding:12px 16px; min-width:150px; }}
    .stat .k {{ color:var(--muted); font-size:14px; }}
    .stat .v {{ font-size:28px; font-weight:700; margin-top:4px; }}
    .stat .hint {{ color:var(--muted); font-size:14px; margin-top:4px; }}
    .bars {{ display:flex; gap:10px; align-items:flex-end; height:64px; padding:0 4px; }}
    .barwrap {{ display:flex; flex-direction:column; align-items:center; gap:4px; min-width:56px; }}
    .bar {{ width:32px; border-radius:4px 4px 0 0; background:#334155; }}
    .bar.g0 {{ background:var(--good); }}
    .bar.g1 {{ background:var(--ok); }}
    .bar.g2 {{ background:#fb923c; }}
    .bar.g3 {{ background:var(--bad); }}
    .barwrap span {{ font-size:13px; color:var(--muted); }}
    .legend {{ display:flex; gap:14px; flex-wrap:wrap; margin-top:10px; font-size:15px; color:var(--muted); }}
    .dot {{ display:inline-block; width:10px; height:10px; border-radius:50%; margin-right:6px; }}
    .layout {{ display:grid; grid-template-columns:280px 1fr; min-height:calc(100vh - 200px); }}
    nav {{ border-right:1px solid var(--line); padding:12px; overflow:auto; }}
    .navhead {{ display:flex; justify-content:space-between; align-items:center; margin-bottom:10px; gap:6px; }}
    .navhead span {{ font-size:15px; color:var(--muted); }}
    .sort {{ background:#1f2a3d; color:var(--text); border:1px solid var(--line); border-radius:6px; padding:6px 10px; cursor:pointer; font-size:14px; }}
    nav button.item {{ display:flex; width:100%; align-items:center; justify-content:space-between; gap:10px;
      background:transparent; color:var(--text); border:1px solid transparent; border-radius:8px;
      padding:10px 12px; margin-bottom:6px; cursor:pointer; font-size:16px; }}
    nav button.item.active {{ background:#1f2a3d; border-color:var(--accent); }}
    nav button.item .id {{ font-variant-numeric:tabular-nums; font-size:16px; }}
    .badge {{ font-variant-numeric:tabular-nums; font-size:16px; font-weight:700; padding:3px 10px; border-radius:999px; }}
    .badge.good {{ color:#052e1c; background:var(--good); }}
    .badge.ok {{ color:#422006; background:var(--ok); }}
    .badge.bad {{ color:#450a0a; background:var(--bad); }}
    .badge.na {{ color:var(--muted); background:#1f2430; }}
    main {{ padding:14px 18px 28px; }}
    .now {{ margin:0 0 12px; font-size:18px; color:var(--muted); }}
    .now b {{ color:var(--text); }}
    .grid {{ display:grid; grid-template-columns:1fr 1fr 1fr; gap:12px; }}
    .stage {{ background:var(--card); border:1px solid var(--line); border-radius:12px; padding:10px; }}
    .stage h2 {{ margin:0 0 8px; font-size:16px; }}
    .stage img {{ width:100%; background:#000; border-radius:8px; display:block; }}
  </style>
</head>
<body>
  <header>
    <h1>비커스 학습 입출력</h1>
    <div class="sub" id="ckptline">한 화면에 전 단계를 보여줍니다. 왼쪽 숫자는 대각선 오차입니다.</div>
    <div class="stats" id="stats"></div>
    <div class="legend">
      <span><i class="dot" style="background:var(--good)"></i>좋음 &lt;3%</span>
      <span><i class="dot" style="background:var(--ok)"></i>보통 3–8%</span>
      <span><i class="dot" style="background:var(--bad)"></i>나쁨 ≥8%</span>
      <span>경도는 대각선 제곱에 반비례해서, 오차 8%면 경도는 대략 16% 흔들립니다.</span>
    </div>
  </header>
  <div class="layout">
    <nav>
      <div class="navhead">
        <span id="navcount">샘플</span>
        <button class="sort" id="sortbtn" type="button">오차큰순</button>
      </div>
      <div id="thumbs"></div>
    </nav>
    <main>
      <div class="now" id="now"></div>
      <div class="grid" id="stages"></div>
    </main>
  </div>
  <script>
    const DATA = {payload};
    let idx = 0;
    let order = DATA.items.map((_, i) => i);
    let worstFirst = false;
    function pct(x) {{
      if (x === null || x === undefined) return "—";
      return (100 * x).toFixed(1) + "%";
    }}
    function diceTxt(x) {{
      if (x === null || x === undefined) return "—";
      return x.toFixed(3);
    }}
    function renderStats() {{
      const s = DATA.summary;
      const ck = s.ckpt || {{}};
      const bits = [];
      if (ck.epoch != null) bits.push("epoch " + ck.epoch);
      if (ck.val_dice != null) bits.push("학습 val Dice " + Number(ck.val_dice).toFixed(3));
      if (ck.size != null) bits.push(ck.size + "px");
      const subset = s.subset || "val";
      const subsetKo = subset === "val" ? "검증 30%" : (subset === "train" ? "학습 70%" : "전체");
      bits.push(subsetKo + " " + DATA.items.length + "장");
      if (s.train_n && s.val_n) bits.push("분할 " + s.train_n + "/" + s.val_n);
      document.getElementById("ckptline").textContent = bits.join(" · ");
      const mean = pct(s.mean_diag);
      const med = pct(s.median_diag);
      const hist = s.hist || [];
      const maxn = Math.max(1, ...hist.map(h => h.n));
      const bars = hist.map((h, i) => {{
        const hpx = Math.max(4, Math.round(44 * h.n / maxn));
        return `<div class="barwrap"><div class="bar g${{i}}" style="height:${{hpx}}px"></div><span>${{h.label}} ${{h.n}}</span></div>`;
      }}).join("");
      document.getElementById("stats").innerHTML = `
        <div class="stat"><div class="k">평균 대각선 오차</div><div class="v">${{mean}}</div><div class="hint">중간값 ${{med}}</div></div>
        <div class="stat"><div class="k">평균 Dice</div><div class="v">${{diceTxt(s.mean_dice)}}</div><div class="hint">마스크가 정답과 겹친 정도</div></div>
        <div class="stat"><div class="k">장 수</div><div class="v">${{s.n_good}} / ${{s.n_ok}} / ${{s.n_bad}}</div><div class="hint">좋음 / 보통 / 나쁨</div></div>
        <div class="stat"><div class="k">오차 분포</div><div class="bars">${{bars}}</div></div>
      `;
    }}
    function renderNav() {{
      const nav = document.getElementById("thumbs");
      nav.innerHTML = "";
      document.getElementById("navcount").textContent = DATA.items.length + "장";
      document.getElementById("sortbtn").textContent = worstFirst ? "번호순" : "오차큰순";
      order.forEach((i) => {{
        const it = DATA.items[i];
        const b = document.createElement("button");
        b.type = "button";
        b.className = "item" + (i === idx ? " active" : "");
        const g = it.grade || "na";
        b.innerHTML = `<span class="id">${{it.id}}</span><span class="badge ${{g}}">${{pct(it.diag)}}</span>`;
        b.onclick = () => {{ idx = i; render(); }};
        nav.appendChild(b);
      }});
    }}
    function renderMain() {{
      const it = DATA.items[idx];
      const now = document.getElementById("now");
      now.innerHTML = `<b>${{it.id}}</b> · 대각선 오차 <b>${{pct(it.diag)}}</b> · Dice ${{diceTxt(it.dice)}}`;
      const box = document.getElementById("stages");
      box.innerHTML = "";
      DATA.steps.forEach(([file, title]) => {{
        const el = document.createElement("section");
        el.className = "stage";
        el.innerHTML = `<h2>${{title}}</h2><img src="samples/${{it.id}}/${{file}}?v=${{DATA.bust}}" alt="${{title}}" />`;
        box.appendChild(el);
      }});
    }}
    function applySort() {{
      order = DATA.items.map((_, i) => i);
      if (worstFirst) {{
        order.sort((a, b) => {{
          const da = DATA.items[a].diag, db = DATA.items[b].diag;
          if (da == null && db == null) return 0;
          if (da == null) return 1;
          if (db == null) return -1;
          return db - da;
        }});
      }}
    }}
    function render() {{
      renderStats();
      renderNav();
      renderMain();
    }}
    document.getElementById("sortbtn").onclick = () => {{
      worstFirst = !worstFirst;
      applySort();
      render();
    }};
    document.addEventListener("keydown", (e) => {{
      const pos = order.indexOf(idx);
      if (e.key === "ArrowDown" || e.key === "ArrowRight") {{
        idx = order[Math.min(order.length - 1, pos + 1)];
        render();
      }}
      if (e.key === "ArrowUp" || e.key === "ArrowLeft") {{
        idx = order[Math.max(0, pos - 1)];
        render();
      }}
    }});
    applySort();
    render();
  </script>
</body>
</html>
"""
    (out / "index.html").write_text(html, encoding="utf-8")


def serve_preview(directory: Path, port: int) -> None:
    class Server(socketserver.ThreadingTCPServer):
        allow_reuse_address = True
        daemon_threads = True

    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(directory), **kwargs)

        def log_message(self, format: str, *args) -> None:
            return

    httpd = None
    bound = port
    for candidate in range(port, port + 15):
        try:
            httpd = Server(("127.0.0.1", candidate), Handler)
            bound = candidate
            break
        except OSError:
            continue
    html = directory / "index.html"
    if httpd is None:
        print(f"포트를 열 수 없습니다. 파일을 직접 엽니다: {html}", flush=True)
        webbrowser.open(html.resolve().as_uri())
        return
    url = f"http://127.0.0.1:{bound}/"
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    time.sleep(0.3)
    print(f"미리보기 사이트: {url}", flush=True)
    print("브라우저가 안 열리면 위 주소를 주소창에 붙여 넣으세요. 끄려면 Ctrl+C", flush=True)
    opened = webbrowser.open(url, new=2)
    if not opened:
        print("브라우저 자동 실행에 실패했습니다. 주소를 직접 여세요.", flush=True)
    try:
        while thread.is_alive():
            thread.join(timeout=0.5)
    except KeyboardInterrupt:
        print("\n미리보기 종료", flush=True)
        httpd.shutdown()


def main() -> None:
    args = parse_args()
    if args.ckpt is None:
        for name in ("best_I.pt", "best_II.pt", "best.pt"):
            cand = ROOT / "weights" / name
            if cand.exists() and cand.stat().st_size > 1024:
                args.ckpt = cand
                break
        if args.ckpt is None:
            args.ckpt = ROOT / "weights" / "best_I.pt"
    data = find_data_root(args.data)
    print(f"data={data}", flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = None
    size = 384
    kind = "mask1"
    meta: dict | None = None
    if args.ckpt and args.ckpt.exists():
        model, size, kind, meta = load_model(args.ckpt, device)
        print(
            f"ckpt loaded: {args.ckpt}  size={size}  device={device}  kind={kind}  "
            f"epoch={meta.get('epoch')}  val_dice={meta.get('val_dice')}",
            flush=True,
        )
    elif args.ckpt:
        print(f"ckpt 없음, 정답만 표시: {args.ckpt}", flush=True)

    ids, train_n, val_n = pick_ids(data, args.split, args.samples, args.subset, args.seed)
    if meta is None:
        meta = {}
    else:
        meta = dict(meta)
    meta["_subset"] = args.subset
    meta["_train_n"] = train_n
    meta["_val_n"] = val_n
    print(
        f"subset={args.subset}  train={train_n}  val={val_n}  showing={len(ids)}  "
        f"seed={args.seed}",
        flush=True,
    )
    print(f"{len(ids)}장 미리보기 생성 중... 다 끝난 뒤에 사이트가 열립니다.", flush=True)
    img_dir, mask_dir = db_dirs(data, args.split)
    items: list[dict] = []
    for stem in ids:
        gray = load_gray(img_dir / f"{stem}.png")
        gt = load_mask(mask_dir / f"{stem}.png")
        pred = None
        pred_verts = None
        if model is not None:
            pred = predict_mask(model, gray, size, device)
            pred_verts = vertices_from_mask(pred)
        gt_verts = vertices_from_mask(gt)
        metrics = sample_metrics(gt, pred, gt_verts, pred_verts)
        dest = args.out / "samples" / stem
        save_steps(
            stem,
            gray,
            gt,
            dest,
            pred,
            gt_verts=gt_verts,
            pred_verts=pred_verts,
            metrics=metrics,
        )
        row = {"id": stem, "has_pred": pred is not None, **metrics}
        items.append(row)
        diag_txt = f"{100 * metrics['diag']:.1f}%" if metrics["diag"] is not None else "n/a"
        dice_txt = f"{metrics['dice']:.3f}" if metrics["dice"] is not None else "n/a"
        print(f"wrote {dest}  diag={diag_txt}  dice={dice_txt}", flush=True)

    summary = summarize(items, meta)
    if summary["mean_diag"] is not None:
        print(
            f"mean diag={100 * summary['mean_diag']:.2f}%  median={100 * summary['median_diag']:.2f}%  "
            f"mean dice={summary['mean_dice']:.3f}  good/ok/bad={summary['n_good']}/{summary['n_ok']}/{summary['n_bad']}",
            flush=True,
        )
    bust = str(int(time.time()))
    write_html(args.out, items, has_model=model is not None, summary=summary, bust=bust)
    html_path = (args.out / "index.html").resolve()
    print(f"파일: {html_path}", flush=True)
    if args.no_open:
        return
    serve_preview(args.out, args.port)


if __name__ == "__main__":
    main()
