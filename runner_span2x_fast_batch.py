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
# 모델 로더
# ==============================================================
def load_model_auto(model_path, device):
    ext = os.path.splitext(model_path)[1].lower()
    if ext in [".pth", ".pt", ".safetensors"]:
        print(f"[model] Loading Torch model → {model_path}")
        loader = ModelLoader()
        # ✅ 최신 spandrel 대응
        model = loader.load_model(model_path).to(device).eval()
        return model
    elif ext == ".onnx":
        if torch.cuda.is_available():
            providers = ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]
        print(f"[model] Loading ONNX model via onnxruntime → {model_path}")
        session = ort.InferenceSession(model_path, providers=providers)
        if "CPUExecutionProvider" in session.get_providers():
            print("[warn] ⚠️ Running on CPU (CUDA/TensorRT not available)")
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
# 단일 프레임 → 텐서 변환
# ==============================================================
def frame_to_tensor(frame, pad):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(rgb).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    return t, pad_frame(frame, pad)

def pad_frame(frame, pad):
    return np.pad(frame, ((pad, pad), (pad, pad), (0, 0)), mode="edge")

# ==============================================================
# 비디오 업스케일 (터보 버전)
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

    # 기존 결과 삭제
    if os.path.exists(output_video):
        os.remove(output_video)

    model = load_model_auto(model_path, device)
    if not isinstance(model, ort.InferenceSession) and use_fp16 and device == "cuda":
        try:
            model = model.half()
            print("[precision] FP16 enabled")
        except Exception as e:
            print(f"[precision] FP16 requested but not applied: {e}")

    # 입력 비디오
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

    # 배치 자동 설정
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

    fourcc = cv2.VideoWriter_fourcc(*codec)
    out = cv2.VideoWriter(output_video, fourcc, fps, (w * 2, h * 2))
    if not out.isOpened():
        raise RuntimeError("❌ Failed to open VideoWriter.")

    is_onnx = isinstance(model, ort.InferenceSession)
    if is_onnx:
        input_name = model.get_inputs()[0].name
        output_name = model.get_outputs()[0].name

    pbar = tqdm(total=total)
    batch_frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        batch_frames.append(frame)

        if len(batch_frames) == batch_size:
            results = process_batch(model, batch_frames, pad, use_fp16, is_onnx,
                                    input_name if is_onnx else None,
                                    output_name if is_onnx else None)
            for img in results:
                out.write(img)
            pbar.update(min(batch_size, total - pbar.n))
            batch_frames = []

    if len(batch_frames) > 0:
        results = process_batch(model, batch_frames, pad, use_fp16, is_onnx,
                                input_name if is_onnx else None,
                                output_name if is_onnx else None)
        for img in results:
            out.write(img)
        pbar.update(min(len(batch_frames), total - pbar.n))

    pbar.close()
    cap.release()
    out.release()
    print(f"✅ Done: {output_video}")

# ==============================================================
# 배치 추론 처리
# ==============================================================
def process_batch(model, frames, pad, use_fp16, is_onnx, input_name, output_name):
    imgs = []
    if is_onnx:
        batch = []
        for f in frames:
            f_pad = np.pad(f, ((pad, pad), (pad, pad), (0, 0)), mode="edge")
            rgb = cv2.cvtColor(f_pad, cv2.COLOR_BGR2RGB)
            arr = rgb.astype(np.float32).transpose(2, 0, 1)[None, ...] / 255.0
            batch.append(arr)
        batch_np = np.concatenate(batch, axis=0)
        result = model.run([output_name], {input_name: batch_np})[0]
        for out in result:
            out_rgb = np.clip(out.transpose(1, 2, 0) * 255, 0, 255).astype(np.uint8)
            cropped = out_rgb[pad:-pad or None, pad:-pad or None]
            imgs.append(cv2.cvtColor(cropped, cv2.COLOR_RGB2BGR))
    else:
        for f in frames:
            rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            t = torch.from_numpy(rgb).float().permute(2, 0, 1).unsqueeze(0) / 255.0
            if use_fp16:
                t = t.half()
            t = t.to(next(model.parameters()).device)
            with torch.no_grad():
                out = model(t)[0].clamp(0, 1)
            out_rgb = (out.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            imgs.append(cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR))
    return imgs

# ==============================================================
# CLI
# ==============================================================
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model",  required=True, help="Model path (.onnx / .pth / .safetensors)")
    p.add_argument("--input",  required=True, help="Input video")
    p.add_argument("--output", required=True, help="Output video")
    p.add_argument("--tile", type=int, default=None)
    p.add_argument("--pad", type=int, default=16)
    p.add_argument("--fp16", action="store_true")
    p.add_argument("--codec", type=str, default="mp4v")
    p.add_argument("--no-tf32", action="store_true")
    p.add_argument("--batch", type=int, default=None)
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
