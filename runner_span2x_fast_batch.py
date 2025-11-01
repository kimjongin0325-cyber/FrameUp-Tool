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

# ==============================================================
# 모델 로더 (ONNX / PTH / safetensors 자동 인식)
# ==============================================================
def load_model_auto(model_path, device):
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

# ==============================================================
# VRAM 기반 타일 자동 조정
# ==============================================================
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

# ==============================================================
# 개별 타일 처리 함수
# ==============================================================
def process_tile(model, tile_bgr, pad, use_fp16=False):
    rgb = cv2.cvtColor(tile_bgr, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(rgb).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    if use_fp16:
        t = t.half()

    if isinstance(model, ort.InferenceSession):
        # ONNX 모델은 float32만 허용
        input_name = model.get_inputs()[0].name
        output_name = model.get_outputs()[0].name
        t_np = t.cpu().numpy().astype(np.float32)
        result = model.run([output_name], {input_name: t_np})[0]
        out = torch.from_numpy(result[0])
    else:
        t = t.to(next(model.parameters()).device)
        with torch.no_grad():
            out = model(t)[0].clamp(0, 1)

    out_rgb = (out.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    cropped = out_rgb[pad*2:-pad*2 or None, pad*2:-pad*2 or None]
    return cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR)

# ==============================================================
# 비디오 업스케일 (배치 처리 적용)
# ==============================================================
def upscale_video(model_path, input_video, output_video,
                  force_tile=None, pad=16, use_fp16=False,
                  codec='mp4v', tf32=True, batch_size=None):
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

    # ✅ 기존 결과 자동 삭제
    if os.path.exists(output_video):
        os.remove(output_video)

    # 모델 로드
    model = load_model_auto(model_path, device)
    if not isinstance(model, ort.InferenceSession) and use_fp16 and device == "cuda":
        try:
            model = model.half()
            print("[precision] FP16 enabled")
        except Exception as e:
            print(f"[precision] FP16 requested but not applied: {e}")

    # 비디오 로드
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

    # VRAM 기반 batch 자동 설정
    if batch_size is None:
        gpus = GPUtil.getGPUs()
        free_mem = gpus[0].memoryFree if gpus else 8000
        if free_mem > 12000:
            batch_size = 6
        elif free_mem > 8000:
            batch_size = 4
        else:
            batch_size = 2
    print(f"[batch] size={batch_size}")

    print(f"[upscale] {w}x{h} -> {w*2}x{h*2}, {fps:.1f}fps")
    print(f"[tiling]  auto={tile}x{tile}, pad={pad}px (force_tile={force_tile})")
    print(f"[frames]  {total} frames")
    print(f"[codec]   fourcc={codec}")

    # 비디오 라이터
    fourcc = cv2.VideoWriter_fourcc(*codec)
    out = cv2.VideoWriter(output_video, fourcc, fps, (w * 2, h * 2))
    if not out.isOpened():
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_video, fourcc, fps, (w * 2, h * 2))
    if not out.isOpened():
        raise RuntimeError("Failed to open VideoWriter. Try --codec avc1 or mp4v.")

    # 메인 루프
    frames_batch = []
    pbar = tqdm(total=total)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames_batch.append(frame)

        if len(frames_batch) == batch_size:
            for f in frames_batch:
                rows_up = []
                for ty in range(tile):
                    row_tiles = []
                    for tx in range(tile):
                        sy, sx = ty * th, tx * tw
                        ey = h if ty == tile - 1 else sy + th
                        ex = w if tx == tile - 1 else sx + tw
                        tile_bgr = np.pad(f[sy:ey, sx:ex], ((pad, pad), (pad, pad), (0, 0)), mode="edge")
                        row_tiles.append(process_tile(model, tile_bgr, pad, use_fp16))
                    rows_up.append(np.hstack(row_tiles))
                out.write(np.vstack(rows_up))
            frames_batch = []
            pbar.update(batch_size)
    pbar.close()

    cap.release()
    out.release()
    print(f"✅ Done: {output_video}")

# ==============================================================
# CLI
# ==============================================================
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",  required=True, help="Path to model file (.onnx / .pth / .safetensors)")
    p.add_argument("--input",  required=True, help="Input video path")
    p.add_argument("--output", required=True, help="Output video path")
    p.add_argument("--tile", type=int, default=None, help="Force tile count per axis")
    p.add_argument("--pad", type=int, default=16, help="Pad size per side")
    p.add_argument("--fp16", action="store_true", help="Enable FP16 inference when supported")
    p.add_argument("--codec", type=str, default="mp4v", help="FourCC for VideoWriter (e.g. avc1, mp4v, H264)")
    p.add_argument("--no-tf32", action="store_true", help="Disable TF32 matmul (Ampere+)")
    p.add_argument("--batch", type=int, default=None, help="Manual batch size override")
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
        tf32=not args.no_tf32,
        batch_size=args.batch
    )

if __name__ == "__main__":
    main()
