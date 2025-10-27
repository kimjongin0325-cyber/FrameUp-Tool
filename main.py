#!/usr/bin/env python3
import os
import cv2
import torch
import argparse
import numpy as np
from spandrel import ModelLoader
from tqdm import tqdm

def choose_model(model_dir="model"):
    priority = [
        "2xNomosUni_span_multijpg.safetensors",
        "2xNomosUni_span_multijpg.pth",
        "2xNomosUni_span_multijpg_fp16_opset17.onnx",
        "2xNomosUni_span_multijpg_fp32_opset17.onnx",
    ]
    for name in priority:
        path = os.path.join(model_dir, name)
        if os.path.exists(path):
            print(f"✅ Using Model: {path}")
            return path
    raise FileNotFoundError(f"No model found in {model_dir}/")

def process_tile(model, tile_bgr):
    tile_rgb = cv2.cvtColor(tile_bgr, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(tile_rgb).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    t = t.to(model.device)
    with torch.no_grad():
        out = model(t)[0].clamp(0, 1)
    out_rgb = (out.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    return cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)

def upscale_2x(model_path, input_path, output_path, tile=2, pad=16):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    loader = ModelLoader()
    mdl = loader.load_from_file(model_path, device=device).eval()

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w*2, h*2))
    tw, th = w // tile, h // tile

    print(f"🎥 {w}x{h} -> {w*2}x{h*2} @ {fps:.1f}fps | tiles={tile}x{tile}, pad={pad}")
    for _ in tqdm(range(total)):
        ret, frame = cap.read()
        if not ret:
            break
        tiles = []
        for ty in range(tile):
            rows = []
            sy = ty * th
            ey = h if ty == tile - 1 else sy + th
            for tx in range(tile):
                sx = tx * tw
                ex = w if tx == tile - 1 else sx + tw
                timg = frame[sy:ey, sx:ex]
                padded = np.pad(timg, ((pad, pad), (pad, pad), (0, 0)), mode="edge")
                up_padded = process_tile(mdl, padded)
                up = up_padded[pad*2: -pad*2 if pad>0 else None, pad*2: -pad*2 if pad>0 else None]
                rows.append(up)
            tiles.append(np.hstack(rows))
        out.write(np.vstack(tiles))

    cap.release()
    out.release()
    print(f"✅ Done: {output_path}")

def main():
    ap = argparse.ArgumentParser(description="2x video upscaler (auto model, tiling)")
    ap.add_argument("--input",  required=True)
    ap.add_argument("--output", default="output_x2.mp4")
    ap.add_argument("--model_dir", default="model")
    ap.add_argument("--tile", type=int, default=2)
    ap.add_argument("--pad",  type=int, default=16)
    args = ap.parse_args()

    model_path = choose_model(args.model_dir)
    upscale_2x(model_path, args.input, args.output, args.tile, args.pad)

if __name__ == "__main__":
    main()
