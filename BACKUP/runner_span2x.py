#!/usr/bin/env python3
import os
import cv2
import torch
import argparse
import numpy as np
from spandrel import ModelLoader
from tqdm import tqdm

def process_tile(model, tile_bgr: np.ndarray) -> np.ndarray:
    """Upscale a single padded tile (BGR uint8) -> returns 2x tile (BGR uint8)."""
    tile_rgb = cv2.cvtColor(tile_bgr, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(tile_rgb).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    t = t.to(model.device)
    with torch.no_grad():
        out = model(t)[0].clamp(0, 1)
    out_rgb = (out.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    return cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)

def upscale_video(model_path: str, input_video: str, output_video: str, tile: int, pad: int):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[runner] device = {device}")

    # (권장) 파편화 방지
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    model = ModelLoader().load_from_file(model_path, device=device).eval()

    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open: {input_video}")

    fps   = cap.get(cv2.CAP_PROP_FPS)
    w     = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h     = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    out = cv2.VideoWriter(output_video,
                          cv2.VideoWriter_fourcc(*"mp4v"),
                          fps, (w * 2, h * 2))

    tw, th = w // tile, h // tile
    print(f"[upscale]  {w}x{h} -> {w*2}x{h*2}, {fps:.1f} FPS")
    print(f"[tiling]   {tile}x{tile} tiles, pad={pad}px")
    print(f"[frames]   {total} frames (may show ? if unknown)")

    for _ in tqdm(range(total)):
        ret, frame = cap.read()
        if not ret:
            break

        # 타일 분할 + 패딩 → 타일별 2x 업스케일 → 재조립
        tiles_up = []
        for ty in range(tile):
            row_tiles = []
            sy = ty * th
            ey = h if ty == tile - 1 else sy + th
            for tx in range(tile):
                sx = tx * tw
                ex = w if tx == tile - 1 else sx + tw
                tile_bgr = frame[sy:ey, sx:ex]

                # 경계 이음새 방지용 패딩
                padded = np.pad(tile_bgr, ((pad, pad), (pad, pad), (0, 0)), mode='edge')
                up_padded = process_tile(model, padded)

                # 업스케일 후 패딩 제거 (2x 되어 pad도 2배)
                up = up_padded[pad*2: -pad*2 if pad > 0 else None,
                               pad*2: -pad*2 if pad > 0 else None]
                row_tiles.append(up)
            tiles_up.append(np.hstack(row_tiles))
        up_frame = np.vstack(tiles_up)
        out.write(up_frame)

    cap.release()
    out.release()
    print("✅ 2x upscaling done:", output_video)

def main():
    p = argparse.ArgumentParser(description="2x video upscaler (tiling to avoid OOM)")
    p.add_argument("--model",  required=True, help="Model path (model/*.pth|*.safetensors|*.onnx)")
    p.add_argument("--input",  required=True, help="Input video")
    p.add_argument("--output", required=True, help="Output video (mp4v, 8bit)")
    p.add_argument("--tile",   type=int, default=2, help="Tile split (NxN). Recommended 2~4")
    p.add_argument("--pad",    type=int, default=16, help="Overlap pad (px). Recommended 8~32")
    args = p.parse_args()
    upscale_video(args.model, args.input, args.output, args.tile, args.pad)

if __name__ == "__main__":
    main()
