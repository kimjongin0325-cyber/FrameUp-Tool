#!/bin/bash
# ============================================================
#  upscale_engine_v2.sh (최적화 버전)
#  - runner_span2x_fast_batch.py 사용
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
# if [ -f "$OUTPUT_VIDEO" ]; then
#   rm "$OUTPUT_VIDEO"
# fi
if [ -f "$AUDIO_FILE" ]; then
  rm "$AUDIO_FILE"
fi

echo "✅ 완료: $FINAL_OUTPUT"
