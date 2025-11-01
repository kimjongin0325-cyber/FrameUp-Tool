#!/usr/bin/env python3
import os
import cv2
import torch
import argparse
import numpy as np
from spandrel import ModelLoader
from tqdm import tqdm
import GPUtil
import onnxruntime as ort

def load_model_auto(model_path, device):
    """
    Auto-select model loader based on file extension (.onnx, .pth, .safetensors)
    """
    ext = os.path.splitext(model_path)[1].lower()
    if ext in [".pth", ".pt", ".safetensors"]:
        print(f"[model] Loading Torch model → {model_path}")
        loader = ModelLoader()
        model = loader.load(model_path).to(device).eval()
        return model
    elif ext == ".onnx":
        print(f"[model] Loading ONNX model via onnxruntime → {model_path}")
        providers = ["CUDAExecutionProvider"] if torch.cuda.is_available() else ["CPUExecutionProvider"]
        session = ort.InferenceSession(model_path, providers=providers)
        return session
    else:
        raise ValueError(f"❌ Unsupported model format: {ext}")

def aggressive_auto_tile(w, h, prefer_full_frame=True):
    gpus = GPUtil.getGPUs()
    if not gpus:
        return 2
    free = gpus[0].memoryFree
    size_factor = (w * h) / (1920 * 1080)
    if free > 12000 and size_factor < 8:
        return 1 if prefer_full_frame else 2
    elif free > 8000:
        return 2
    elif free > 4000:
        return 3
    else:
        return 4

def process_tile(model, tile_bgr, pad, use_fp16=False):
    rgb = cv2.cvtColor(tile_bgr, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(rgb).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    if use_fp16:
        t = t.half()

    if isinstance(model, ort.InferenceSession):
        # ONNX Runtime 추론
        input_name = model.get_inputs()[0].name
        output_name = model.get_outputs()[0].name
        result = model.run([output_name], {input_name: t.cpu().numpy()})[0]
        out = torch.from_numpy(result[0])
    else:
        # Torch 모델 추론
        t = t.to(next(model.parameters()).device)
        with torch.no_grad():
            out = model(t)[0].clamp(0, 1)

    out_rgb = (out.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    cropped = out_rgb[pad*2:-pad*2 or None, pad*2:-pad*2 or None]
    return cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR)

def upscale_video(model_path, input_video, output_video, force_tile=None, pad=16, use_fp16=False, codec='mp4v', tf32=True):
    torch.backends.cudnn.benchmark = True
    if tf32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision('medium')
        except Exception:
            pass

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[runner] device = {device}")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    # ✅ 모델 자동 로드
    model = load_model_auto(model_path, device)
    if not isinstance(model, ort.InferenceSession) and use_fp16 and device == "cuda":
        try:
            model = model.half()
            print("[precision] FP16 enabled")
        except Exception as e:
            print(f"[precision] FP16 requested but not applied: {e}")

    cap = cv2.VideoCapture(input_video, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open: {input_video}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    tile = force_tile if force_tile is not None else aggressive_auto_tile(w, h, prefer_full_frame=True)
    tw, th = w // tile, h // tile

    print(f"[upscale] {w}x{h} -> {w*2}x{h*2}, {fps:.1f}fps")
    print(f"[tiling]  auto={tile}x{tile}, pad={pad}px (force_tile={force_tile})")
    print(f"[frames]  {total} frames")
    print(f"[codec]   fourcc={codec}")

    fourcc = cv2.VideoWriter_fourcc(*codec)
    out = cv2.VideoWriter(output_video, fourcc, fps, (w * 2, h * 2))
    if not out.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_video, fourcc, fps, (w * 2, h * 2))
    if not out.isOpened():
        raise RuntimeError("Failed to open VideoWriter. Try --codec avc1 or mp4v.")

    pbar = tqdm(total=total)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        rows_up = []
        for ty in range(tile):
            row_tiles = []
            for tx in range(tile):
                sy, sx = ty * th, tx * tw
                ey = h if ty == tile - 1 else sy + th
                ex = w if tx == tile - 1 else sx + tw
                tile_bgr = np.pad(frame[sy:ey, sx:ex], ((pad, pad), (pad, pad), (0, 0)), mode="edge")
                row_tiles.append(process_tile(model, tile_bgr, pad, use_fp16))
            rows_up.append(np.hstack(row_tiles))
        up = np.vstack(rows_up)
        out.write(up)
        pbar.update(1)
    pbar.close()

    cap.release()
    out.release()
    print(f"✅ Done: {output_video}")

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",  required=True, help="Path to model file (.pth / .onnx / .safetensors)")
    p.add_argument("--input",  required=True, help="Input video path")
    p.add_argument("--output", required=True, help="Output video path")
    p.add_argument("--tile", type=int, default=None, help="Force tile count per axis")
    p.add_argument("--pad", type=int, default=16, help="Pad size per side")
    p.add_argument("--fp16", action="store_true", help="Enable FP16 (half) inference when supported")
    p.add_argument("--codec", type=str, default="mp4v", help="FourCC for VideoWriter (e.g. avc1, mp4v, H264)")
    p.add_argument("--no-tf32", action="store_true", help="Disable TF32 matmul (Ampere+)")
    return p.parse_args()

def main():
    args = parse_args()
    upscale_video(
        model_path=args.model,
        input_video=args.input,
        output_video=args.output,
        force_tile=args.tile,
        pad=args.pad,
        use_fp16=args.fp16,
        codec=args.codec,
        tf32=not args.no_tf32
    )

if __name__ == "__main__":
    main()
