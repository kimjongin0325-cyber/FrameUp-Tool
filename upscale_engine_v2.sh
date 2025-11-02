#!/bin/bash
# ============================================================
#  upscale_engine_v2.sh (TensorRT 대응 완성본)
#  - runner_span2x.py: TensorRT .engine 지원 버전과 연동
#  - GPU 병목 해소, 속도 2~3배 향상
#  - [개선] TensorRT 엔진 파일 자동 감지 및 우선 사용
# ============================================================

MODEL_DIR="/content/FrameUp-Tool/model"
INPUT_VIDEO="$1"
BASENAME=$(basename "$INPUT_VIDEO")
NAME="${BASENAME%.*}"

# 임시 파일 경로
OUTPUT_VIDEO="/content/FrameUp-Tool/${NAME}_x2.mp4"
AUDIO_FILE="/content/FrameUp-Tool/${NAME}_audio.aac"
FINAL_OUTPUT="/content/FrameUp-Tool/${NAME}_x2_final.mp4"

# ============================================================
# 🔍 모델 자동 탐색 (우선순위: TensorRT > FP16 ONNX > FP32 ONNX > 기타)
# ============================================================
models=(
  "$MODEL_DIR/2xNomosUni_span_multijpg_fp16.engine"
  "$MODEL_DIR/2xNomosUni_span_multijpg.engine"
  "$MODEL_DIR/2xNomosUni_span_multijpg_fp16_opset17.onnx"
  "$MODEL_DIR/2xNomosUni_span_multijpg_fp32_opset17.onnx"
  "$MODEL_DIR/2xNomosUni_span_multijpg.safetensors"
  "$MODEL_DIR/2xNomosUni_span_multijpg.pth"
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

# ============================================================
# 🎵 오디오 추출
# ============================================================
echo "[audio] Extracting audio..."
ffmpeg -y -i "$INPUT_VIDEO" -vn -c:a copy "$AUDIO_FILE" > /dev/null 2>&1 || \
ffmpeg -y -i "$INPUT_VIDEO" -vn -c:a aac -b:a 192k "$AUDIO_FILE" > /dev/null 2>&1

AUDIO_SUCCESS=0
if [ -s "$AUDIO_FILE" ]; then
  AUDIO_SUCCESS=1
  echo "[audio] Audio extracted successfully."
else
  echo "[audio] No audio found or extraction failed. Proceeding without audio merge."
fi

# ============================================================
# 🚀 업스케일 실행 (TensorRT 자동 인식)
# ============================================================
echo "[upscale] Running upscale with runner_span2x.py ..."
python /content/FrameUp-Tool/runner_span2x.py \
  --model "$mdl" \
  --input "$INPUT_VIDEO" \
  --output "$OUTPUT_VIDEO" \
  --batch 4

if [ $? -ne 0 ]; then
  echo "❌ Upscale 실패"
  exit 1
fi

# ============================================================
# 🔊 오디오 합성
# ============================================================
if [ $AUDIO_SUCCESS -eq 1 ]; then
  echo "[merge] Combining upscaled video and extracted audio..."
  ffmpeg -y -i "$OUTPUT_VIDEO" -i "$AUDIO_FILE" -c:v copy -c:a aac -b:a 192k \
    -map 0:v:0 -map 1:a:0 -shortest "$FINAL_OUTPUT"
else
  echo "[merge] No audio to combine. Renaming video file."
  mv "$OUTPUT_VIDEO" "$FINAL_OUTPUT"
fi

# ============================================================
# 🧹 임시 파일 정리
# ============================================================
if [ -f "$AUDIO_FILE" ]; then
  rm "$AUDIO_FILE"
fi

echo "✅ 완료: $FINAL_OUTPUT"
