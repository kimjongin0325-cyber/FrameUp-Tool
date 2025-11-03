import cv2, os, numpy as np, time
import pycuda.driver as cuda
import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

cuda.init()
device = cuda.Device(0)
ctx = device.make_context()

engine_path = "/content/drive/MyDrive/2x_fp16.plan"
with open(engine_path, "rb") as f:
    runtime = trt.Runtime(TRT_LOGGER)
    engine = runtime.deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()

print("✅ TensorRT 엔진 로드 완료")

input_path = "/content/m.mp4"
output_path = "/content/upscaled_m_trt.mp4"
cap = cv2.VideoCapture(input_path)
fps = cap.get(cv2.CAP_PROP_FPS)
w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_path, fourcc, fps, (w*2, h*2))

input_name = "input"
output_name = "output"
context.set_input_shape(input_name, (1, 3, h, w))
output_shape = context.get_tensor_shape(output_name)

frame_count = 0
t0 = time.time()
while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (w, h))
    input_data = np.transpose(frame, (2, 0, 1))[None, :, :, :].astype(np.float16) / 255.0

    d_input = cuda.mem_alloc(input_data.nbytes)
    d_output = cuda.mem_alloc(int(np.prod(output_shape) * np.dtype(np.float16).itemsize))
    cuda.memcpy_htod(d_input, input_data)

    context.set_tensor_address(input_name, int(d_input))
    context.set_tensor_address(output_name, int(d_output))
    stream = cuda.Stream()
    context.execute_async_v3(stream_handle=stream.handle)

    output_data = np.empty(output_shape, dtype=np.float16)
    cuda.memcpy_dtoh_async(output_data, d_output, stream)
    stream.synchronize()

    out_frame = np.clip(output_data[0].transpose(1, 2, 0) * 255.0, 0, 255).astype(np.uint8)
    out_frame = cv2.cvtColor(out_frame, cv2.COLOR_RGB2BGR)
    out.write(out_frame)

    frame_count += 1
    if frame_count % 10 == 0:
        print(f"🎞️ {frame_count} frames done...")

cap.release()
out.release()
ctx.pop()
print(f"✅ 완료! 총 {frame_count} 프레임 ({time.time()-t0:.1f}s)")
print(f"📁 출력파일: {output_path}")
