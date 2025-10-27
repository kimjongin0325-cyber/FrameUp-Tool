#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="model"

models=(
  "$MODEL_DIR/2xNomosUni_span_multijpg.safetensors"
  "$MODEL_DIR/2xNomosUni_span_multijpg.pth"
  "$MODEL_DIR/2xNomosUni_span_multijpg_fp16_opset17.onnx"
  "$MODEL_DIR/2xNomosUni_span_multijpg_fp32_opset17.onnx"
)

choose_model() {
  for m in "${models[@]}"; do
    [[ -f "$m" ]] && echo "$m" && return
  done
  echo "none"
}

extract_audio() {
  ffmpeg -y -i "$1" -vn -c:a copy "$2" || \
  ffmpeg -y -i "$1" -vn -c:a aac -b:a 192k "$2"
}

basename_noext() {
  local f="${1##*/}"
  echo "${f%.*}"
}

run() {
  local in="$1"
  local name; name=$(basename_noext "$in")
  local aud="${name}.aac"
  local out="${name}_x2.mp4"

  mdl=$(choose_model)
  [[ "$mdl" == "none" ]] && {
    echo "❌ No model found in model/"
    exit 1
  }

  echo "[Audio-Mux] -> $aud"
  extract_audio "$in" "$aud"

  echo "[Upscale 2X] -> $out"
  python runner_span2x.py --model "$mdl" --input "$in" --output "$out"

  echo "✅ Output: /content/FrameUp-Tool/$out"
}

run "$1"
