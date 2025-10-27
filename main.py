import os
import cv2
import torch
import argparse
from spandrel import ModelLoader
from tqdm import tqdm

os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"

def choose_model(model_dir="model"):
    """Choose best model available in priority order."""
    priority = [
        "2xNomosUni_span_multijpg.safetensors",
        "2xNomosUni_span_multijpg.pth",
        "2xNomosUni_span_multijpg_fp16_opset17.onnx",
        "2xNomosUni_span_multijpg_fp32_opset17.onnx"
    ]
    for m in priority:
        path = os.path.join(model_dir, m)
        if os.path.exists(path):
            print(f"✅ Using Model: {path}")
            return path
    raise FileNotFoundError("❌ No valid model found in /model")

def upscale_2x(model_path, input_path, output_path):
    print(f"🔍 Loading model: {model_path}")
    loader = ModelLoader()
    mdl = loader.load_from_file(model_path, device="cuda" if torch.cuda.is_available() else "cpu")

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError("❌ Failed to open input video")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width * 2, height * 2))

    print(f"🎥 Upscaling {width}x{height} → {width*2}x{height*2}")
    print(f"📌 Total frames: {total}")

    for _ in tqdm(range(total)):
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        inp = torch.from_numpy(rgb).float().permute(2,0,1).unsqueeze(0).contiguous() / 255.0
        inp = inp.to(mdl.device).to(torch.float32)

        with torch.no_grad():
            out_frame = mdl(inp)[0]

        out_frame = (out_frame.mul(255).clamp(0,255)
                     .permute(1,2,0).byte().cpu().numpy())
        out_frame = cv2.cvtColor(out_frame, cv2.COLOR_RGB2BGR)

        out.write(out_frame)

    cap.release()
    out.release()

def main():
    parser = argparse.ArgumentParser(description="2x Video Upscale Tool")
    parser.add_argument("--input", required=True, help="Input video path")
    parser.add_argument("--output", default="output_x2.mp4", help="Upscaled output path")
    args = parser.parse_args()

    model_path = choose_model("model")
    upscale_2x(model_path, args.input, args.output)

    print("✅ Upscale Done")
    print(f"📁 Saved: {args.output}")

if __name__ == "__main__":
    main()
