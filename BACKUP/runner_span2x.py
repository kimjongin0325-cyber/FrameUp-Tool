#!/usr/bin/env python3
import os
import cv2
import torch
import argparse
import numpy as np
from spandrel import ModelLoader
from tqdm import tqdm
import GPUtil
from torch.cuda.amp import autocast
import kornia.color as K

def auto_tile(w, h):
    gpus = GPUtil.getGPUs()
    if not gpus:
        return 2
    free = gpus[0].memoryFree
    size_factor = (w*h) / (1920*1080)

    if free > 8000 and size_factor < 4:
        return 1
    elif free > 4000:
        return 2
    elif free > 2000:
        return 3
    else:
        return 4

def upscale_video(model_path, input_video, output_video, batch_size):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[runner] device = {device}")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    model = ModelLoader().load_from_file(model_path).to(device).eval()

    # ⚠️ CPU I/O (I/O 병목 현상의 원인)
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
    num_tiles = tile * tile

    print(f"[upscale] {w}x{h} -> {w*2}x{h*2}, {fps:.1f}fps")
    print(f"[tiling]  auto={tile}x{tile}, pad={pad}px, tiles={num_tiles}")
    print(f"[frames]  {total} frames")
    
    out = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*"XVID"), fps, (w*2,h*2))

    for _ in tqdm(range(total)):
        ret, frame = cap.read()
        if not ret:
            break
        
        tile_tensors = []
        
        for ty in range(tile):
            for tx in range(tile):
                sy, sx = ty*th, tx*tw
                ey = h if ty==tile-1 else sy+th
                ex = w if tx==tile-1 else sx+tw
                
                tile_bgr = np.pad(frame[sy:ey, sx:ex], ((pad,pad),(pad,pad),(0,0)), mode="edge")
                
                # CPU -> GPU 데이터 전송
                t = torch.from_numpy(tile_bgr).float().to(device)
                
                # 🌟 GPU에서 색상 변환 및 정규화
                t = t.permute(2,0,1).unsqueeze(0)
                t = K.bgr_to_rgb(t) / 255.0
                
                tile_tensors.append(t)

        # 🌟 GPU 배치 생성
        batch_t = torch.cat(tile_tensors, dim=0)

        with torch.no_grad():
            # 🌟 GPU 연산 및 FP16 (autocast) 사용
            with autocast(enabled=device=="cuda"):
                out_batch = model(batch_t).clamp(0,1)
        
        rows_up = []
        for ty in range(tile):
            row_tiles = []
            for tx in range(tile):
                tile_idx = ty * tile + tx
                out_tile = out_batch[tile_idx]
                
                # 🌟 GPU에서 BGR로 변환
                out_bgr_tensor = K.rgb_to_bgr(out_tile)
                
                # GPU -> CPU 데이터 전송
                out_bgr = (out_bgr_tensor.permute(1,2,0).cpu().numpy()*255).astype(np.uint8)
                
                final_tile = out_bgr[pad*2:-pad*2 or None, pad*2:-pad*2 or None]
                row_tiles.append(final_tile)
                
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
    p.add_argument("--batch", type=int, default=1, help="Frame batch size (maintained for script compatibility).")
    args = p.parse_args()
    upscale_video(args.model, args.input, args.output, args.batch)

if __name__ == "__main__":
    main()
