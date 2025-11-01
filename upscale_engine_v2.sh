#!/bin/bash
# ============================================================
#  upscale_engine_v2.sh (최적화 버전)
#  - runner_span2x_fast_batch.py 사용
#  - GPU 병목 해소, 속도 2~3배 향상
# ============================================================

MODEL_DIR="/content/FrameUp-Tool/model"
INPUT_VIDEO="$1"
BASENAME=$(basename "$INPUT_VIDEO")
NAME="${BASENAME%.*}"
OUTPUT_VIDEO="/content/FrameUp-Tool/${NAME}_x2.mp4"
AUDIO_FILE="/content/FrameUp-Tool/${NAME}_audio.aac"

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
ffmpeg -y -i "$INPUT_VIDEO" -vn -c:a copy "$AUDIO_FILE" || \
ffmpeg -y -i "$INPUT_VIDEO" -vn -c:a aac -b:a 192k "$AUDIO_FILE"

# -----------------------------
# 업스케일 실행 (FAST-BATCH 버전)
# -----------------------------
echo "[upscale] Running upscale with runner_span2x_fast_batch.py ..."
python /content/FrameUp-Tool/runner_span2x_fast_batch.py \
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
echo "[merge] Combining upscaled video and extracted audio..."
ffmpeg -y -i "$OUTPUT_VIDEO" -i "$AUDIO_FILE" -c:v copy -c:a aac -b:a 192k \
  "/content/FrameUp-Tool/${NAME}_x2_final.mp4"

echo "✅ 완료: /content/FrameUp-Tool/${NAME}_x2_final.mp4"
