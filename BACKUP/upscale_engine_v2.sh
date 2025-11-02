
#!/usr/bin/env bash
set -euo pipefail

# -------- Model candidates (priority) --------
MODEL_DIR="model"
MODEL_SAFE="${MODEL_DIR}/2xNomosUni_span_multijpg.safetensors"
MODEL_PTH="${MODEL_DIR}/2xNomosUni_span_multijpg.pth"
MODEL_FP16="${MODEL_DIR}/2xNomosUni_span_m#!/bin/bash
# ============================================================
#  upscale_engine_v2.sh (최적화 버전)
#  - runner_span2x.py 사용으로 변경됨 (배치 기능 유지 여부는 Python 파일에 따름)
#  - GPU 병목 해소, 속도 2~3배 향상
#  - [수정] 오디오 합성 로직 개선
# ============================================================

MODEL_DIR="/content/FrameUp-Tool/model"
INPUT_VIDEO="$1"
BASENAME=$(basename "$INPUT_VIDEO")
NAME="${BASENAME%.*}"

# 임시 파일 경로
OUTPUT_VIDEO="/content/FrameUp-Tool/${NAME}_x2.mp4"
AUDIO_FILE="/content/FrameUp-Tool/${NAME}_audio.aac"
# 최종 출력 파일 경로
FINAL_OUTPUT="/content/FrameUp-Tool/${NAME}_x2_final.mp4"

# -----------------------------
# 모델 자동 탐색
# -----------------------------
models=(
  "$MODEL_DIR/2xNomosUni_span_multijpg.safetensors"
  "$MODEL_DIR/2xNomosUni_span_multijpg.pth"
  "$MODEL_DIR/2xNomosUni_span_multijpg_fp16_opset17.onnx"
  "$MODEL_DIR/2xNomosUni_span_multijpg_fp32_opset17.onnx"
)

mdl=""
for m in "${models[@]}"; do
  if [ -f "$m" ]; then
    mdl="$m"
    break
  fi
done

if [ -z "$mdl" ]; then
  echo "❌ 모델 파일을 찾을 수 없습니다. ($MODEL_DIR)"
  exit 1
fi

echo "[model] Using model: $mdl"

# -----------------------------
# 오디오 추출
# -----------------------------
echo "[audio] Extracting audio..."
# 오디오 추출 시도 (첫 번째 명령어 실패 시 두 번째 명령어 실행)
ffmpeg -y -i "$INPUT_VIDEO" -vn -c:a copy "$AUDIO_FILE" > /dev/null 2>&1 || \
ffmpeg -y -i "$INPUT_VIDEO" -vn -c:a aac -b:a 192k "$AUDIO_FILE" > /dev/null 2>&1

# 오디오 파일이 성공적으로 추출되었는지 확인 (파일 크기가 0보다 큰지)
AUDIO_SUCCESS=0
if [ -s "$AUDIO_FILE" ]; then
  AUDIO_SUCCESS=1
  echo "[audio] Audio extracted successfully."
else
  # 오디오 추출에 실패했거나 오디오가 없는 경우
  echo "[audio] No audio found or extraction failed. Proceeding without audio merge."
fi

# -----------------------------
# 업스케일 실행 (파일 이름 변경 적용)
# -----------------------------
echo "[upscale] Running upscale with runner_span2x.py ..."
# NOTE: 원본 스크립트 (runner_span2x.py)는 --batch 인자를 지원하지 않을 수 있습니다.
# 만약 runner_span2x.py가 배치 처리를 지원하지 않는다면, 이 인자(--batch 4)를 제거하거나,
# runner_span2x.py 파일 내부에 배치 처리 로직을 추가해야 합니다.
python /content/FrameUp-Tool/runner_span2x.py \
  --model "$mdl" \
  --input "$INPUT_VIDEO" \
  --output "$OUTPUT_VIDEO" \
  --batch 4

if [ $? -ne 0 ]; then
  echo "❌ Upscale 실패"
  exit 1
fi

# -----------------------------
# 오디오 합성
# -----------------------------
if [ $AUDIO_SUCCESS -eq 1 ]; then
  echo "[merge] Combining upscaled video and extracted audio..."
  # 오디오가 있는 경우: 비디오와 오디오를 합성
  ffmpeg -y -i "$OUTPUT_VIDEO" -i "$AUDIO_FILE" -c:v copy -c:a aac -b:a 192k \
    -map 0:v:0 -map 1:a:0 -shortest "$FINAL_OUTPUT"
else
  echo "[merge] No audio to combine. Renaming video file."
  # 오디오가 없는 경우: 업스케일된 비디오 파일의 이름을 최종 이름으로 변경
  mv "$OUTPUT_VIDEO" "$FINAL_OUTPUT"
fi

# -----------------------------
# 임시 파일 정리 (선택 사항)
# -----------------------------
if [ -f "$AUDIO_FILE" ]; then
  rm "$AUDIO_FILE"
fi

echo "✅ 완료: $FINAL_OUTPUT"ultijpg_fp16_opset17.onnx"
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
