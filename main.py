import os
import sys
sys.path.append("/content/FrameUp-Tool")

import torch
import cv2
import numpy as np
import time
import shutil
from spandrel import ModelLoader  # SPAN 자동 감지/로드 지원


device = "cuda" if torch.cuda.is_available() else "cpu"


# 🔹 임시파일 및 이전 결과 자동 정리
def clean_temp():
    print("🧹 Cleaning temporary files...")
    temp_files = [
        "/content/audio.m4a",
        "/content/upscaled_x2.mp4",
        "/content/final_output.mp4"
    ]
    for f in temp_files:
        if os.path.exists(f):
            os.remove(f)


# 🔹 SPAN 업스케일 실행
def upscale_video(model_path, input_video, output_video):
    print(f"🚀 Using device: {device}")

    print("🧠 Loading model:", model_path)
    model = ModelLoader().load_from_file(model_path).to(device)
    model.eval()

    print("🎬 Loading:", input_video)
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

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"📈 {w}x{h} → {w*2}x{h*2}, frames: {frame_count}")

    print("⚡ Upscaling...")
    start = time.time()

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
        print(f"{frame_idx}/{frame_count} frames", end="\r")

    cap.release()
    out.release()

    t = time.time() - start
    print(f"\n✅ Upscale Complete → {output_video} (⏱ {t:.1f}s)")


# 🔹 Drive에 자동 백업
def save_to_drive():
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    dst = f"/content/drive/MyDrive/FrameUp/output_{timestamp}.mp4"
    shutil.copy("/content/upscaled_x2.mp4", dst)
    print(f"📁 Saved: {dst}")


def main():
    while True:
        print("\n===============================")
        input_video = input("🎬 업스케일할 영상 경로 입력: ")

        if not os.path.exists(input_video):
            print("❌ 파일 없음 — 경로 다시 확인!")
            continue

        clean_temp()

        model_path = "/content/FrameUp-Tool/models/4xmssim_span_pretrain.pth"
        output_file = "/content/upscaled_x2.mp4"

        upscale_video(model_path, input_video, output_file)

        save_to_drive()

        again = input("\n➕ 다음 영상도 업스케일? (y/n): ").strip().lower()
        if again != "y":
            print("\n👋 작업 종료!")
            break


if __name__ == "__main__":
    main()
