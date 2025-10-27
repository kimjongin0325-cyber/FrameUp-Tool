#!/usr/bin/env python3
import os
import cv2
import torch
import argparse
import numpy as np
from spandrel import ModelLoader
from tqdm import tqdm
import GPUtil

def auto_tile(w, h):
    gpus = GPUtil.getGPUs()
    if not gpus:
        return 2  # CPU fallback
    free = gpus[0].memoryFree  # MB
    size_factor = (w*h) / (1920*1080)

    if free > 8000 and size_factor < 4:
        return 1
    elif free > 4000:
        return 2
    elif free > 2000:
        return 3
    else:
        return 4

def process_tile(model, tile_bgr, pad):
    rgb = cv2.cvtColor(tile_bgr, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(rgb).float().permute(2,0,1).unsqueeze(0)/255.0
    t = t.to(model.device)

    with torch.no_grad():
        out = model(t)[0].clamp(0,1)

    out_rgb = (out.permute(1,2,0).cpu().numpy()*255).astype(np.uint8)
    return cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)[pad*2:-pad*2 or None, pad*2:-pad*2 or None]

def upscale_video(model_path, input_video, output_video):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[runner] device = {device}")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    model = ModelLoader().load_from_file(model_path).to(device).eval()

    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open: {input_video}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    tile = auto_tile(w, h)
    pad = 16
    tw, th = w//tile, h//tile

    print(f"[upscale] {w}x{h} -> {w*2}x{h*2}, {fps:.1f}fps")
    print(f"[tiling]  auto={tile}x{tile}, pad={pad}px")
    print(f"[frames]  {total} frames")

    out = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w*2,h*2))

    for _ in tqdm(range(total)):
        ret, frame = cap.read()
        if not ret:
            break
        rows_up = []
        for ty in range(tile):
            row_tiles = []
            for tx in range(tile):
                sy, sx = ty*th, tx*tw
                ey = h if ty==tile-1 else sy+th
                ex = w if tx==tile-1 else sx+tw
                tile_bgr = np.pad(frame[sy:ey, sx:ex], ((pad,pad),(pad,pad),(0,0)), mode="edge")
                row_tiles.append(process_tile(model, tile_bgr, pad))
            rows_up.append(np.hstack(row_tiles))
        out.write(np.vstack(rows_up))

    cap.release()
    out.release()
    print(f"✅ Done: {output_video}")

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model",  required=True)
    p.add_argument("--input",  required=True)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    upscale_video(args.model, args.input, args.output)

if __name__ == "__main__":
    main()
