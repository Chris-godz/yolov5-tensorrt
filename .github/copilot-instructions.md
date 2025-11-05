# YOLOv5-TensorRT to DEEPX Migration

## Mission
Migrate this YOLOv5 object detection library from NVIDIA TensorRT to DEEPX NPU framework.

## Critical Rules

### 1. Learn DEEPX API First
- Read `dxrtAPI/dxrt/inference_engine.h` to understand `dxrt::InferenceEngine` usage
- Read `dxrtAPI/dxrt/tensor.h` for input/output data handling
- Key pattern: `auto engine = dxrt::InferenceEngine(modelPath); auto outputs = engine.Run(inputPtr);`

### 2. Replace TensorRT Components

**Engine Loading:**
```cpp
// BEFORE: TensorRT
_runtime = nvinfer1::createInferRuntime(_trtLogger);
_engine = _runtime->deserializeCudaEngine(buffer, size);
_context = _engine->createExecutionContext();

// AFTER: DEEPX
_inferenceEngine = std::make_unique<dxrt::InferenceEngine>(enginePath);
```

**Inference Execution:**
```cpp
// BEFORE: TensorRT
_context->enqueueV2(_bindingMemory.data(), _cudaStream, nullptr);
cudaStreamSynchronize(_cudaStream);

// AFTER: DEEPX  
dxrt::TensorPtrs outputs = _inferenceEngine->Run(inputPtr);
float* outputData = (float*)outputs[0]->data();
```

### 3. Remove ALL CUDA Code
- Delete: `#include <cuda_runtime_api.h>`, `#include <NvInfer.h>`
- Delete: `cudaMalloc`, `cudaFree`, `cudaMemcpy`, `cudaStream_t`
- Delete: `CvCudaPreprocessor` class and all `cv::cuda::*` operations
- Delete: `YOLOV5_OPENCV_HAS_CUDA` conditional compilation blocks

### 4. Keep CPU Preprocessing Only
Keep the `CvCpuPreprocessor` class - it does letterbox resize, normalization, and BGR→planar conversion using OpenCV CPU functions.

### 5. Update Build System
In `CMakeLists.txt`:
- Remove: `find_package(CUDA)` and `find_package(TensorRT)`  
- Add: `include_directories(dxrtAPI)`
- Change file extensions: `.engine` → `.dxnn`

### 6. Preserve Public API
Keep the same interface in `yolov5_detector.hpp`:
```cpp
detector.init();
detector.loadEngine("model.dxnn");  // Just change extension
detector.detect(image, &detections);
```

## Key Files to Modify
1. `src/yolov5_detector_internal.cpp` - Remove CUDA, keep CvCpuPreprocessor
2. `include/yolov5_detector_internal.hpp` - Replace TensorRT headers with DEEPX
3. `src/yolov5_detector.cpp` - Use dxrt::InferenceEngine for inference
4. `src/yolov5_builder.cpp` - Validate .dxnn files instead of building engines
5. `CMakeLists.txt` - Update dependencies

## Implementation Notes
- Input format: NCHW planar, float32, normalized [0,1]
- DEEPX models (.dxnn) are pre-compiled offline
- No CUDA streams needed - DEEPX handles device scheduling
- Keep existing Logger and Result error codes
