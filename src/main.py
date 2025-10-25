import os
import argparse
import subprocess
import torch
from PIL import Image
import numpy as np
import cv2
import time

# 모델 경로 설정
model_dir = os.path.join(os.path.dirname(__file__), "..", "model")
model_path = os.path.join(model_dir, "4xmssim_span_pretrain.pth")

# GPU 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Span 기반 업스케일 모델 로더 (예시)
class UpscaleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.load(model_path, map_location=device)
        self.model.to(device)
        self.model.eval()

    def forward(self, img):
        with torch.no_grad():
            img = img.to(device)
            output = self.model(img)
        return output

def upscale_frame(model, frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = torch.from_numpy(frame / 255.0).permute(2, 0, 1).float().unsqueeze(0)
    frame = model(frame)[0].permute(1, 2, 0).cpu().numpy()
    frame = (frame * 255.0).clip(0, 255).astype(np.uint8)
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

def upscale_video(input_path):
    print(f"[INFO] Loading video: {input_path}")
    cap = cv2.VideoCapture(input_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    out_w, out_h = width * 2, height * 2  # x4 업스케일 → px2 px2
    print(f"[INFO] Resolution: {width}x{height} → {out_w}x{out_h}")

    output_path = input_path.replace(".mp4", "_4x.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (out_w, out_h))

    model = UpscaleModel()

    start = time.time()

    for idx in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break
        upscaled = upscale_frame(model, frame)
        writer.write(upscaled)

        print(f"[PROCESS] {idx+1}/{frame_count} frames", end="\r")

    writer.release()
    cap.release()

    end = time.time()
    print(f"\n✅ DONE: {output_path}")
    print(f"⏱ Total time: {end-start:.1f} seconds")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="input video path")
    args = parser.parse_args()

    upscale_video(args.input)

if __name__ == "__main__":
    main()

