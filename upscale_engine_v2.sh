!cat <<'EOF' > upscale_engine_v2.sh
#!/usr/bin/env bash
set -euo pipefail

MODEL_FP16="model/2xNomosUni_span_multijpg_fp16_opset17.onnx"
MODEL_FP32="model/2xNomosUni_span_multijpg_fp32_opset17.onnx"
MODEL_SAFE="model/2xNomosUni_span_multijpg.safetensors"
MODEL_PTH ="model/2xNomosUni_span_multijpg.pth"

# ONNX 대신 torch+spandrel 경로 고정: safetensors > pth > fp16 onnx > fp32 onnx
choose_model(){
  [[ -f "$MODEL_SAFE" ]] && { echo "$MODEL_SAFE"; return; }
  [[ -f "$MODEL_PTH"  ]] && { echo "$MODEL_PTH";  return; }
  [[ -f "$MODEL_FP16" ]] && { echo "$MODEL_FP16"; return; }
  [[ -f "$MODEL_FP32" ]] && { echo "$MODEL_FP32"; return; }
  echo "none"
}

extract_audio(){ ffmpeg -y -i "$1" -vn -acodec copy "$2"; }
final_encode(){ ffmpeg -y -i "$1" -i "$2" -c:v libx265 -pix_fmt yuv420p10le -x265-params crf=20:preset=medium -c:a copy -movflags +faststart "$3"; }
basename_noext(){ local f="$1"; f="${f##*/}"; echo "${f%.*}"; }

run_upscale_2x(){
  local in="$1" out="$2" mdl
  mdl=$(choose_model)
  [[ "$mdl" == "none" ]] && { echo "ERROR: no model in model/"; exit 1; }
  python runner_span2x.py --model "$mdl" --input "$in" --output "$out"
}

main_loop(){
  while true; do
    read -rp "Input video path: " INPUT
    [[ -f "$INPUT" ]] || { echo "File not found: $INPUT"; continue; }

    local NAME; NAME=$(basename_noext "$INPUT")
    local A="${NAME}.aac" P1="${NAME}_x2.mp4" P2="${NAME}_x4.mp4" FINAL="${NAME}_4k10bit.mp4"

    echo "[Audio] -> $A"; extract_audio "$INPUT" "$A"
    echo "[Pass1] -> $P1"; run_upscale_2x "$INPUT" "$P1"

    read -rp "Continue to Pass2? (y/n): " go2
    if [[ "$go2" != "y" ]]; then
      echo "[Final] -> $FINAL"; final_encode "$P1" "$A" "$FINAL"
      read -rp "Next file? (y/n): " nxt; [[ "$nxt" == "y" ]] && continue || { echo "Done"; break; }
    fi

    echo "[Pass2] -> $P2"; run_upscale_2x "$P1" "$P2"
    echo "[Final] -> $FINAL"; final_encode "$P2" "$A" "$FINAL"
    read -rp "Next file? (y/n): " nxt; [[ "$nxt" == "y" ]] && continue || { echo "Done"; break; }
  done
}
main_loop
EOF

!chmod +x upscale_engine_v2.sh
!./upscale_engine_v2.sh

