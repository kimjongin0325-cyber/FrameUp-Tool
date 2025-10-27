%cd /content/FrameUp-Tool
!cat <<'EOF' > runner_span2x.py
import argparse, os, time
import cv2, torch, numpy as np
from spandrel import ModelLoader  # SPAN/ESRGAN loader

def upscale_video(model_path, input_video, output_video):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[runner] device = {device}")
    model = ModelLoader().load_from_file(model_path).to(device).eval()

    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open: {input_video}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    out = cv2.VideoWriter(
        output_video,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w*2, h*2)
    )

    print(f"[upscale]  {w}x{h} -> {w*2}x{h*2}, {fps} FPS")
    print(f"[frames]   {total} frames (may show ? if unknown)")
    t0 = time.time()

    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok: break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(rgb/255.0).permute(2,0,1).float().unsqueeze(0).to(device)

        with torch.no_grad():
            up = model(t).clamp(0,1)[0].permute(1,2,0).cpu().numpy()

        bgr = cv2.cvtColor((up*255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        out.write(bgr)

        idx += 1
        if idx % 30 == 0:
            print(f"[progress] {idx}/{total}", end="\r")

    cap.release()
    out.release()
    print(f"\n[done] {output_video} ({time.time()-t0:.1f}s)")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="2x Span Upscaler Runner")
    ap.add_argument("--model", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    upscale_video(args.model, args.input, args.output)
EOF

!ls -al runner_span2x.py
