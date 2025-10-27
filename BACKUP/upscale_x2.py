import sys
sys.path.append("/content/FrameUp-Tool")

import torch
import cv2
import numpy as np
from spandrel import ModelLoader

model_path = "/content/FrameUp-Tool/model/2xNomosUni_span_multijpg.pth"
input_path = "/content/drive/MyDrive/m.mp4"  # ✅ 구글 드라이브 경로
output_path = "/content/upscaled_x2.mp4"

device = "cuda" if torch.cuda.is_available() else "cpu"

print("🚀 Loading SPAN model...")
model = ModelLoader().load_from_file(model_path).to(device)
model.eval()

cap = cv2.VideoCapture(input_path)
fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w*2, h*2))

print("⚡ Upscaling...")
while True:
    ret, frame = cap.read()
    if not ret: break
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(rgb / 255.0).permute(2,0,1).float().unsqueeze(0).to(device)

    with torch.no_grad():
        up = model(t).clamp(0,1)[0].permute(1,2,0).cpu().numpy()

    bgr = cv2.cvtColor((up*255).astype(np.uint8), cv2.COLOR_RGB2BGR)
    out.write(bgr)

cap.release()
out.release()

print("✅ Upscale Complete →", output_path)
