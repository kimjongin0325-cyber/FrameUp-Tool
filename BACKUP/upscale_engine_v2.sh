
#!/usr/bin/env bash
set -euo pipefail

# -------- Model candidates (priority) --------
MODEL_DIR="model"
MODEL_SAFE="${MODEL_DIR}/2xNomosUni_span_multijpg.safetensors"
MODEL_PTH="${MODEL_DIR}/2xNomosUni_span_multijpg.pth"
MODEL_FP16="${MODEL_DIR}/2xNomosUni_span_multijpg_fp16_opset17.onnx"
MODEL_FP32="${MODEL_DIR}/2xNomosUni_span_multijpg_fp32_opset17.onnx"

choose_model() {
  [[ -f "$MODEL_SAFE" ]] && { echo "$MODEL_SAFE"; return; }
  [[ -f "$MODEL_PTH"  ]] && { echo "$MODEL_PTH";  return; }
  [[ -f "$MODEL_FP16" ]] && { echo "$MODEL_FP16"; return; }
  [[ -f "$MODEL_FP32" ]] && { echo "$MODEL_FP32"; return; }
  echo "none"
}

# -------- Utils --------
extract_audio() {  # copy 실패 시 AAC 재인코드 폴백
  ffmpeg -y -i "$1" -vn -c:a copy "$2" || ffmpeg -y -i "$1" -vn -c:a aac -b:a 192k "$2"
}

final_encode() {  # x265 10bit 최종 인코딩 (CRF 20, preset medium)
  ffmpeg -y -i "$1" -i "$2" \
    -c:v libx265 -pix_fmt yuv420p10le -x265-params crf=20:preset=medium \
    -c:a copy -movflags +faststart "$3"
}

basename_noext() { local f="$1"; f="${f##*/}"; echo "${f%.*}"; }

run_upscale_2x() {
  local in="$1" out="$2" mdl tile="${3:-2}" pad="${4:-16}"
  mdl=$(choose_model)
  [[ "$mdl" == "none" ]] && { echo "ERROR: no model in ${MODEL_DIR}/"; exit 1; }
  # 파편화 방지 권장 옵션
  export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
  python runner_span2x.py --model "$mdl" --input "$in" --output "$out" --tile "$tile" --pad "$pad"
}

# -------- Main interactive loop --------
main_loop() {
  while true; do
    read -rp "Input video path: " INPUT
    [[ -f "$INPUT" ]] || { echo "File not found: $INPUT"; continue; }

    local NAME; NAME=$(basename_noext "$INPUT")
    local A="${NAME}.aac"
    local P1="${NAME}_x2.mp4"
    local P2="${NAME}_x4.mp4"
    local FINAL="${NAME}_4k10bit.mp4"

    echo "[Audio] -> $A";  extract_audio "$INPUT" "$A"
    echo "[Pass1] -> $P1"; run_upscale_2x "$INPUT" "$P1" 2 16

    read -rp "Continue to Pass2? (y/n): " go2
    if [[ "${go2,,}" != "y" ]]; then
      echo "[Final] -> $FINAL"; final_encode "$P1" "$A" "$FINAL"
      read -rp "Next file? (y/n): " nxt; [[ "${nxt,,}" == "y" ]] && continue || { echo "Done"; break; }
    fi

    echo "[Pass2] -> $P2"; run_upscale_2x "$P1" "$P2" 2 16
    echo "[Final] -> $FINAL"; final_encode "$P2" "$A" "$FINAL"
    read -rp "Next file? (y/n): " nxt; [[ "${nxt,,}" == "y" ]] && continue || { echo "Done"; break; }
  done
}

main_loop
