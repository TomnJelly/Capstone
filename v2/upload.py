#!/usr/bin/env python3
"""
사진만 올리면 예측 마스크·꼭짓점을 보여주는 로컬 사이트.
정답 마스크는 없습니다. 기존 미리보기와 같은 U-Net + 에지 서브픽셀입니다.
"""

from __future__ import annotations

import argparse
import base64
import http.server
import json
import re
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
    diagonals_px,
    draw_geometry,
    overlay_mask,
    predict_mask,
    vertices_from_mask,
)

MAX_BYTES = 25 * 1024 * 1024
ALLOWED_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Upload a photo, see predicted indent")
    p.add_argument("--port", type=int, default=8767)
    p.add_argument("--no-open", action="store_true")
    return p.parse_args()


def load_model(ckpt_path: Path, device: torch.device) -> tuple[nn.Module, int, dict]:
    blob = torch.load(ckpt_path, map_location=device, weights_only=False)
    size = int(blob.get("size", 384))
    out_ch = int(blob["model"]["out.weight"].shape[0])
    if out_ch != 1:
        raise SystemExit(f"{ckpt_path} 는 1채널 마스크 모델이 아닙니다 (out_ch={out_ch}).")
    base = int(blob.get("base", 48))
    model: nn.Module = UNet(base=base)
    model = model.to(device)
    model.load_state_dict(blob["model"])
    model.eval()
    meta = {
        "epoch": blob.get("epoch"),
        "val_dice": blob.get("val_dice"),
        "arch": blob.get("arch") or "unet1_mask384_v2",
        "size": size,
        "db": blob.get("db"),
        "path": str(ckpt_path),
    }
    return model, size, meta


def png_data_url(img: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise RuntimeError("png encode failed")
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def decode_gray(raw: bytes) -> np.ndarray:
    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("이미지를 읽지 못했습니다. png/jpg 인지 확인하세요.")
    return img


def _header_map(blob: bytes) -> dict[str, str]:
    text = blob.decode("utf-8", "replace")
    out: dict[str, str] = {}
    for line in text.split("\r\n"):
        if ":" not in line:
            continue
        key, val = line.split(":", 1)
        out[key.strip().lower()] = val.strip()
    return out


def _disp_name(disp: str) -> tuple[str | None, str | None]:
    name = None
    filename = None
    m = re.search(r'name="([^"]*)"', disp)
    if m:
        name = m.group(1)
    star = re.search(r"filename\*=(?:UTF-8|utf-8)''([^;]+)", disp)
    if star:
        from urllib.parse import unquote

        filename = unquote(star.group(1))
    else:
        m = re.search(r'filename="([^"]*)"', disp)
        if m:
            filename = m.group(1)
        else:
            m = re.search(r"filename=([^;]+)", disp)
            if m:
                filename = m.group(1).strip()
    return name, filename


def parse_multipart(content_type: str, body: bytes) -> tuple[dict[str, str], list[tuple[str, bytes]]]:
    if "boundary=" not in content_type:
        raise ValueError("multipart boundary 없음")
    boundary = content_type.split("boundary=", 1)[1].strip().strip('"').encode("ascii", "ignore")
    marker = b"--" + boundary
    fields: dict[str, str] = {}
    files: list[tuple[str, bytes]] = []
    for chunk in body.split(marker):
        if not chunk or chunk in (b"--", b"--\r\n"):
            continue
        if chunk.startswith(b"--"):
            continue
        if chunk.startswith(b"\r\n"):
            chunk = chunk[2:]
        if chunk.endswith(b"\r\n"):
            chunk = chunk[:-2]
        split_at = chunk.find(b"\r\n\r\n")
        if split_at < 0:
            continue
        headers = _header_map(chunk[:split_at])
        data = chunk[split_at + 4 :]
        disp = headers.get("content-disposition", "")
        name, filename = _disp_name(disp)
        if not name:
            continue
        if filename is not None:
            files.append((filename, data))
        else:
            fields[name] = data.decode("utf-8", "replace")
    return fields, files


PAGE = r"""<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>비커스 예측 (업로드)</title>
  <style>
    :root { --bg:#0e1117; --card:#171b22; --line:#2a3140; --text:#e8edf5; --muted:#9aa6b8; --accent:#3b82f6; }
    * { box-sizing: border-box; }
    body { margin:0; font-family:"Segoe UI",sans-serif; background:var(--bg); color:var(--text); }
    header { padding:14px 20px 12px; border-bottom:1px solid var(--line); }
    h1 { margin:0 0 6px; font-size:24px; }
    .sub { color:var(--muted); font-size:16px; }
    .row { display:flex; flex-wrap:wrap; gap:10px; align-items:center; margin-top:12px; }
    .drop {
      border:1px dashed #3d4a63; border-radius:12px; padding:18px 16px; min-width:280px; flex:1;
      background:#12161d; color:var(--muted); cursor:pointer; text-align:center;
    }
    .drop.over { border-color:var(--accent); color:var(--text); }
    select, button {
      background:#1f2a3d; color:var(--text); border:1px solid var(--line);
      border-radius:8px; padding:10px 14px; font-size:15px;
    }
    button.run { background:var(--accent); border-color:#2563eb; cursor:pointer; font-weight:600; }
    button.run:disabled { opacity:0.5; cursor:wait; }
    .err { color:#f87171; margin-top:10px; min-height:1.2em; }
    .layout { display:grid; grid-template-columns:260px 1fr; min-height:calc(100vh - 170px); }
    nav { border-right:1px solid var(--line); padding:12px; overflow:auto; }
    nav button.item {
      display:flex; width:100%; align-items:center; justify-content:space-between; gap:8px;
      background:transparent; color:var(--text); border:1px solid transparent; border-radius:8px;
      padding:10px 12px; margin-bottom:6px; cursor:pointer; font-size:15px; text-align:left;
    }
    nav button.item.active { background:#1f2a3d; border-color:var(--accent); }
    main { padding:14px 18px 28px; }
    .now { margin:0 0 12px; font-size:18px; color:var(--muted); }
    .now b { color:var(--text); }
    .grid { display:grid; grid-template-columns:1fr 1fr 1fr; gap:12px; }
    .stage { background:var(--card); border:1px solid var(--line); border-radius:12px; padding:10px; }
    .stage h2 { margin:0 0 8px; font-size:16px; }
    .stage img { width:100%; background:#000; border-radius:8px; display:block; cursor:zoom-in; }
    .zoom { position:fixed; inset:0; display:none; align-items:center; justify-content:center;
      flex-direction:column; gap:10px; background:rgba(0,0,0,.88); z-index:80; padding:20px; cursor:zoom-out; }
    .zoom.on { display:flex; }
    .zoom img { max-width:min(96vw, 1600px); max-height:calc(100vh - 72px); object-fit:contain;
      background:#000; border-radius:8px; cursor:zoom-out; }
    .zoom .cap { color:var(--text); font-size:16px; }
    .empty { color:var(--muted); padding:40px 8px; }
    @media (max-width: 980px) {
      .layout { grid-template-columns:1fr; }
      .grid { grid-template-columns:1fr; }
    }
  </style>
</head>
<body>
  <header>
    <h1>비커스 예측</h1>
    <div class="sub">사진을 올리면 압흔 마스크와 꼭짓점만 뽑습니다. 정답 비교는 없습니다. 결과 사진을 클릭하면 확대됩니다.</div>
    <div class="row">
      <div class="drop" id="drop"><span id="droptext">여기로 사진을 끌어오거나 클릭해서 고르세요</span>
        <input id="files" type="file" accept="image/png,image/jpeg,image/bmp,image/tiff,image/webp" multiple hidden />
      </div>
      <select id="split" title="가중치">
        <option value="I">모델 I (best_I.pt)</option>
        <option value="II">모델 II (best_II.pt)</option>
      </select>
      <button class="run" id="run" type="button">예측</button>
    </div>
    <div class="err" id="err"></div>
  </header>
  <div class="layout">
    <nav id="thumbs"><div class="empty">아직 올린 사진이 없습니다.</div></nav>
    <main>
      <div class="now" id="now"></div>
      <div class="grid" id="stages"></div>
    </main>
  </div>
  <div class="zoom" id="zoom">
    <div class="cap" id="zoomcap"></div>
    <img id="zoomimg" alt="" />
  </div>
  <script>
    const filesEl = document.getElementById("files");
    const drop = document.getElementById("drop");
    const dropText = document.getElementById("droptext");
    const runBtn = document.getElementById("run");
    const errEl = document.getElementById("err");
    const splitEl = document.getElementById("split");
    let picked = [];
    let items = [];
    let idx = 0;
    const zoomEl = document.getElementById("zoom");
    const zoomImg = document.getElementById("zoomimg");
    const zoomCap = document.getElementById("zoomcap");
    let zoomOpen = false;
    function closeZoom() {
      zoomOpen = false;
      zoomEl.classList.remove("on");
    }
    function toggleZoom(src, title) {
      if (zoomOpen && zoomImg.getAttribute("data-src") === src) {
        closeZoom();
        return;
      }
      zoomOpen = true;
      zoomEl.classList.add("on");
      zoomImg.src = src;
      zoomImg.setAttribute("data-src", src);
      zoomCap.textContent = title + " · 다시 클릭하면 닫힘";
    }
    zoomEl.addEventListener("click", () => closeZoom());
    document.addEventListener("keydown", (e) => {
      if (e.key === "Escape" && zoomOpen) closeZoom();
    });

    function setErr(msg) { errEl.textContent = msg || ""; }
    function setPicked(list) {
      picked = [...list].filter((f) => f && ((f.type || "").startsWith("image/") || /\.(png|jpe?g|bmp|tiff?|webp)$/i.test(f.name)));
      dropText.textContent = picked.length
        ? (picked.length + "장 선택됨 · 다시 고르려면 클릭")
        : "여기로 사진을 끌어오거나 클릭해서 고르세요";
    }
    drop.onclick = () => filesEl.click();
    drop.ondragover = (e) => { e.preventDefault(); drop.classList.add("over"); };
    drop.ondragleave = () => drop.classList.remove("over");
    drop.ondrop = (e) => {
      e.preventDefault();
      drop.classList.remove("over");
      setPicked(e.dataTransfer.files);
    };
    filesEl.onchange = () => setPicked(filesEl.files);

    function render() {
      const nav = document.getElementById("thumbs");
      const now = document.getElementById("now");
      const box = document.getElementById("stages");
      if (!items.length) {
        nav.innerHTML = '<div class="empty">아직 올린 사진이 없습니다.</div>';
        now.textContent = "";
        box.innerHTML = "";
        return;
      }
      nav.innerHTML = "";
      items.forEach((it, i) => {
        const b = document.createElement("button");
        b.type = "button";
        b.className = "item" + (i === idx ? " active" : "");
        const mean = it.mean == null ? "—" : it.mean.toFixed(1) + "px";
        b.innerHTML = `<span>${it.name}</span><span>${mean}</span>`;
        b.onclick = () => { idx = i; render(); };
        nav.appendChild(b);
      });
      const it = items[idx];
      const d1 = it.d1 == null ? "—" : it.d1.toFixed(1);
      const d2 = it.d2 == null ? "—" : it.d2.toFixed(1);
      const mean = it.mean == null ? "—" : it.mean.toFixed(1);
      now.innerHTML = `<b>${it.name}</b> · d1 <b>${d1}px</b> · d2 <b>${d2}px</b> · 평균 <b>${mean}px</b> · ${it.model}`;
      const stages = [
        ["입력 이미지", it.input],
        ["예측 마스크", it.mask],
        ["예측 기하", it.geom],
      ];
      box.innerHTML = "";
      stages.forEach(([title, src]) => {
        const el = document.createElement("section");
        el.className = "stage";
        el.innerHTML = `<h2>${title}</h2><img src="${src}" alt="${title}" />`;
        const img = el.querySelector("img");
        img.onclick = () => toggleZoom(img.src, title);
        box.appendChild(el);
      });
    }

    runBtn.onclick = async () => {
      if (!picked.length) { setErr("사진을 먼저 고르세요."); return; }
      setErr("");
      runBtn.disabled = true;
      runBtn.textContent = "예측 중…";
      items = [];
      idx = 0;
      render();
      try {
        for (let i = 0; i < picked.length; i++) {
          runBtn.textContent = `예측 중… ${i + 1}/${picked.length}`;
          const form = new FormData();
          form.append("file", picked[i]);
          form.append("split", splitEl.value);
          const res = await fetch("/predict", { method: "POST", body: form });
          const data = await res.json();
          if (!res.ok || !data.ok) throw new Error(data.error || ("HTTP " + res.status));
          items.push(data);
          idx = items.length - 1;
          render();
        }
      } catch (e) {
        setErr(String(e.message || e));
      } finally {
        runBtn.disabled = false;
        runBtn.textContent = "예측";
      }
    };
    render();
  </script>
</body>
</html>
"""


class Predictor:
    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lock = threading.Lock()
        self.models: dict[str, tuple[nn.Module, int, dict]] = {}
        for tag, name in (("I", "best_I.pt"), ("II", "best_II.pt")):
            path = ROOT / "weights" / name
            if path.exists() and path.stat().st_size > 1024:
                print(f"loading {path} on {self.device} ...", flush=True)
                self.models[tag] = load_model(path, self.device)
        if not self.models:
            raise SystemExit("weights/best_I.pt 또는 best_II.pt 가 필요합니다.")

    def run(self, raw: bytes, filename: str, split: str) -> dict:
        tag = split.upper().replace("DB-", "")
        if tag not in self.models:
            raise ValueError(f"모델 {tag} 가중치가 없습니다. 있는 것: {sorted(self.models)}")
        ext = Path(filename).suffix.lower()
        if ext and ext not in ALLOWED_EXT:
            raise ValueError(f"지원하지 않는 확장자입니다: {ext}")
        gray = decode_gray(raw)
        model, size, meta = self.models[tag]
        with self.lock:
            pred = predict_mask(model, gray, size, self.device)
        verts = vertices_from_mask(pred, gray)
        d1 = d2 = mean = None
        if verts is not None:
            d1, d2, mean = diagonals_px(verts)
        mask_vis = overlay_mask(gray, pred, (231, 76, 60))
        geom = draw_geometry(gray, verts, (0, 70, 255))
        return {
            "ok": True,
            "name": Path(filename).name or "upload",
            "model": f"DB-{tag}  {meta.get('arch')}  size={size}",
            "d1": d1,
            "d2": d2,
            "mean": mean,
            "verts": {k: [float(v[0]), float(v[1])] for k, v in (verts or {}).items()},
            "input": png_data_url(gray),
            "mask": png_data_url(mask_vis),
            "geom": png_data_url(geom),
        }


def make_handler(predictor: Predictor):
    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, format: str, *args) -> None:
            return

        def _send(self, code: int, body: bytes, content_type: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path.split("?", 1)[0] in ("/", "/index.html"):
                self._send(200, PAGE.encode("utf-8"), "text/html; charset=utf-8")
                return
            self._send(404, b"not found", "text/plain; charset=utf-8")

        def do_POST(self) -> None:
            if self.path.split("?", 1)[0] != "/predict":
                self._send(404, b'{"ok":false,"error":"not found"}', "application/json")
                return
            try:
                length = int(self.headers.get("Content-Length") or "0")
                if length <= 0 or length > MAX_BYTES:
                    raise ValueError("파일이 비었거나 너무 큽니다 (최대 25MB).")
                body = self.rfile.read(length)
                fields, files = parse_multipart(self.headers.get("Content-Type") or "", body)
                if not files:
                    raise ValueError("업로드된 파일이 없습니다.")
                filename, raw = files[0]
                split = fields.get("split", "I")
                payload = predictor.run(raw, filename, split)
                raw_json = json.dumps(payload, ensure_ascii=False).encode("utf-8")
                self._send(200, raw_json, "application/json; charset=utf-8")
            except Exception as exc:
                raw_json = json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False).encode("utf-8")
                self._send(400, raw_json, "application/json; charset=utf-8")

    return Handler


def serve(port: int, handler_cls, open_browser: bool) -> None:
    class Server(socketserver.ThreadingTCPServer):
        allow_reuse_address = True
        daemon_threads = True

    httpd = None
    bound = port
    for candidate in range(port, port + 15):
        try:
            httpd = Server(("127.0.0.1", candidate), handler_cls)
            bound = candidate
            break
        except OSError:
            continue
    if httpd is None:
        raise SystemExit("포트를 열 수 없습니다.")
    url = f"http://127.0.0.1:{bound}/"
    print(f"업로드 예측 사이트: {url}", flush=True)
    print("브라우저가 안 열리면 위 주소를 주소창에 붙여 넣으세요. 끄려면 Ctrl+C", flush=True)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    time.sleep(0.3)
    if open_browser:
        opened = webbrowser.open(url, new=2)
        if not opened:
            print("브라우저 자동 실행에 실패했습니다. 주소를 직접 여세요.", flush=True)
    try:
        while thread.is_alive():
            thread.join(timeout=0.5)
    except KeyboardInterrupt:
        print("\n업로드 사이트 종료", flush=True)
        httpd.shutdown()


def main() -> None:
    args = parse_args()
    predictor = Predictor()
    print(f"models={list(predictor.models)}  device={predictor.device}", flush=True)
    serve(args.port, make_handler(predictor), open_browser=not args.no_open)


if __name__ == "__main__":
    main()
