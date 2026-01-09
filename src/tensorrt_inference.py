
import numpy as np
import cv2
from typing import Tuple, List, Optional
import os

try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    TRT_AVAILABLE = True
except ImportError as e:
    TRT_AVAILABLE = False
    print(f"TensorRT Python API not found: {e}")


class TensorRTInference:
    
    def __init__(self, engine_path: str):
        if not TRT_AVAILABLE:
            raise RuntimeError("TensorRT Python API required!")
        
        if not os.path.exists(engine_path):
            raise FileNotFoundError(f"Engine file not found: {engine_path}")
        
        self.engine_path = engine_path
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.engine = None
        self.context = None
        self.inputs = []
        self.outputs = []
        self.bindings = []
        self.stream = cuda.Stream()

        self.letterbox_scale = 1.0
        self.letterbox_pad = (0, 0)

        self._load_engine()
    
    def _load_engine(self):
        try:
            with open(self.engine_path, 'rb') as f:
                runtime = trt.Runtime(self.logger)
                engine_data = f.read()
                self.engine = runtime.deserialize_cuda_engine(engine_data)
            
            if self.engine is None:
                raise RuntimeError(
                    f"Failed to load engine: {self.engine_path}\n"
                    "Possibly created with older TensorRT version.\n"
                    "Solution: Delete and rebuild engine file:\n"
                    "  docker exec basketball_deepstream python3 /app/engine_check/convert_to_engine.py"
                )
            
            self.context = self.engine.create_execution_context()
        except Exception as e:
            error_msg = str(e)
            if "old deserialization" in error_msg or "newer plan file" in error_msg:
                raise RuntimeError(
                    f"TensorRT Engine version mismatch!\n"
                    f"   Engine file: {self.engine_path}\n"
                    f"   Error: {error_msg}\n\n"
                    f"Solution: Delete and rebuild engine file:\n"
                    f"   1. docker exec -it basketball_deepstream bash\n"
                    f"   2. rm {self.engine_path}\n"
                    f"   3. python3 /app/engine_check/convert_to_engine.py\n"
                    f"   OR automatically:\n"
                    f"   docker exec basketball_deepstream bash -c 'rm {self.engine_path} && python3 /app/engine_check/convert_to_engine.py'"
                ) from e
            raise
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            shape = self.engine.get_tensor_shape(name)
            size = trt.volume(shape)

            device_mem = cuda.mem_alloc(size * dtype().itemsize)
            self.context.set_tensor_address(name, int(device_mem))

            mode = self.engine.get_tensor_mode(name)
            if mode == trt.TensorIOMode.INPUT:
                self.inputs.append({
                    'name': name,
                    'dtype': dtype,
                    'shape': shape,
                    'device_mem': device_mem
                })
            else:
                self.outputs.append({
                    'name': name,
                    'dtype': dtype,
                    'shape': shape,
                    'device_mem': device_mem
                })
    
    def infer(self, image: np.ndarray) -> List[np.ndarray]:
        input_data = self._preprocess(image)

        cuda.memcpy_htod_async(
            self.inputs[0]['device_mem'],
            input_data,
            self.stream
        )

        self.context.execute_async_v3(stream_handle=self.stream.handle)

        outputs = []
        for output in self.outputs:
            host_mem = np.empty(output['shape'], dtype=output['dtype'])
            cuda.memcpy_dtoh_async(host_mem, output['device_mem'], self.stream)
            outputs.append(host_mem)

        self.stream.synchronize()
        return outputs
    
    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        img = self._letterbox(image, new_shape=(640, 640))

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = img.astype(np.float32) / 255.0

        img = img.transpose(2, 0, 1)

        img = np.expand_dims(img, axis=0)

        img = np.ascontiguousarray(img)

        return img

    def _letterbox(self, img: np.ndarray, new_shape=(640, 640), color=(114, 114, 114)):
        shape = img.shape[:2]

        r = min(new_shape[0] / shape[1], new_shape[1] / shape[0])

        new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
        dw, dh = new_shape[0] - new_unpad[0], new_shape[1] - new_unpad[1]

        dw /= 2
        dh /= 2

        self.letterbox_scale = r
        self.letterbox_pad = (dw, dh)

        if shape[::-1] != new_unpad:
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)

        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

        return img
    
    def __del__(self):
        if hasattr(self, 'context') and self.context is not None:
            del self.context
        if hasattr(self, 'engine') and self.engine is not None:
            del self.engine


def create_tensorrt_inference(engine_path: str) -> Optional[TensorRTInference]:
    try:
        return TensorRTInference(engine_path)
    except Exception as e:
        print(f"Failed to create TensorRT inference: {e}")
        return None


