import sys
sys.path.append("/content/FrameUp-Tool")

import argparse
import torch
import cv2
import numpy as np
from spandrel import ModelLoader
import os

def upscale_video(model_path, input_video, output_video):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Using device: {device}")

    print("🧠 Loading model:", model_path)
    model = ModelLoader().load_from_file(model_path).to(device)
    model.eval()

    print("🎬 Loading video:", input_video)
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        raise FileNotFoundError("❌ Video cannot be opened")

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(
        output_video,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w * 2, h * 2)
    )

    print("⚡ Upscaling starts...")
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(rgb / 255.0).permute(2,0,1).float().unsqueeze(0).to(device)

        with torch.no_grad():
            up = model(t).clamp(0,1)[0].permute(1,2,0).cpu().numpy()

        bgr = cv2.cvtColor((up * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        out.write(bgr)

        frame_idx += 1
        print(f"{frame_idx} frames processed", end="\r")

    cap.release()
    out.release()

    print(f"\n✅ Upscale Complete → {output_video}")

def main():
    parser = argparse.ArgumentParser(description="SPAN x2 Upscaler CLI")
    parser.add_argument("--input", required=True, help="Input video path")
    parser.add_argument("--output", required=True, help="Output video path")
    parser.add_argument("--model", required=True, help="Model (.pth) path")
    args = parser.parse_args()

    upscale_video(
        model_path=args.model,
        input_video=args.input,
        output_video=args.output
    )

if __name__ == "__main__":
    main()
