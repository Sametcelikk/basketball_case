
import numpy as np
import cv2
from typing import List, Tuple, Optional
import sys
import os

try:
    from tensorrt_inference import create_tensorrt_inference
    TRT_AVAILABLE = True
except ImportError as e:
    TRT_AVAILABLE = False
    print(f"TensorRT inference not found: {e}")

try:
    import gi
    gi.require_version('Gst', '1.0')
    from gi.repository import Gst, GLib
    import pyds
    DEEPSTREAM_AVAILABLE = True
    print("DeepStream Python bindings loaded")
except ImportError as e:
    DEEPSTREAM_AVAILABLE = False
    print(f"DeepStream Python bindings not found: {e}")
    print("TensorRT Python API will be used")


class DeepStreamInferenceBase:
    
    def __init__(self, config_path: str, model_name: str):
        if not DEEPSTREAM_AVAILABLE:
            raise RuntimeError(
                "DeepStream Python bindings not found! "
                "Install: cd /opt/nvidia/deepstream/deepstream/lib && python3 setup.py install"
            )
        
        self.config_path = config_path
        self.model_name = model_name
        self.is_initialized = False
        
        Gst.init(None)
    
    def check_config(self):
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        print(f"{self.model_name} config loaded: {self.config_path}")


class DeepStreamPoseModel:
    
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.model_name = "Pose Model (TensorRT)"
        
        import sys
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from config import PROJECT_DIR
        
        self.engine_path = os.path.join(PROJECT_DIR, "models/custom-pose-model-2.engine")
        self.confidence_threshold = 0.2
        
        self._load_config(PROJECT_DIR)
        
        if not os.path.exists(self.engine_path):
            print(f"Engine file not found: {self.engine_path}")
            print(f"Please run convert_to_engine.py script")
            self.infer_engine = None
        else:
            self.infer_engine = create_tensorrt_inference(self.engine_path)
            if self.infer_engine:
                print(f"{self.model_name} TensorRT engine loaded")
    
    def _load_config(self, project_dir: str):
        try:
            current_section = None
            with open(self.config_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('['):
                        current_section = line.strip('[]')
                    elif '=' in line and not line.startswith('#'):
                        key, value = line.split('=', 1)
                        key = key.strip()
                        value = value.strip()
                        
                        if current_section == 'model':
                            if key == 'engine-file':
                                self.engine_path = os.path.join(project_dir, value)
                            elif key == 'confidence-threshold':
                                self.confidence_threshold = float(value)
        except Exception as e:
            print(f"Pose config load error: {e}, using defaults")
    
    def __call__(self, frame: np.ndarray, conf: float = None, verbose: bool = False) -> List:
        if conf is None:
            conf = self.confidence_threshold
        if self.infer_engine is None:
            keypoints_xy = np.zeros((0, 33, 2), dtype=np.float32)
            keypoints_conf = np.zeros((0, 33), dtype=np.float32)
            result = PoseResult(keypoints_xy, keypoints_conf)
            return [result]

        outputs = self.infer_engine.infer(frame)

        keypoints_xy, keypoints_conf = self._parse_pose_output(
            outputs,
            conf,
            frame.shape[:2]
        )

        result = PoseResult(keypoints_xy, keypoints_conf)
        return [result]
    
    def _parse_pose_output(
        self,
        outputs: List[np.ndarray],
        conf_threshold: float,
        orig_shape: Tuple[int, int]
    ) -> Tuple[np.ndarray, np.ndarray]:
        NUM_KEYPOINTS = 33
        MODEL_INPUT_SIZE = 640

        if len(outputs) == 0:
            return np.zeros((0, NUM_KEYPOINTS, 2), dtype=np.float32), np.zeros((0, NUM_KEYPOINTS), dtype=np.float32)

        output = outputs[0]

        if len(output.shape) == 3:
            output = output[0].T

        num_features = output.shape[1] if len(output.shape) > 1 else 0

        detected_keypoints = (num_features - 5) // 3 if num_features > 5 else 0

        if detected_keypoints == 0:
            print(f"Unexpected pose output format: shape={output.shape}, features={num_features}")
            return np.zeros((0, NUM_KEYPOINTS, 2), dtype=np.float32), np.zeros((0, NUM_KEYPOINTS), dtype=np.float32)

        conf_mask = output[:, 4] > conf_threshold
        output = output[conf_mask]

        if len(output) == 0:
            return np.zeros((0, NUM_KEYPOINTS, 2), dtype=np.float32), np.zeros((0, NUM_KEYPOINTS), dtype=np.float32)

        best_idx = np.argmax(output[:, 4])
        best_detection = output[best_idx]

        letterbox_scale = getattr(self.infer_engine, 'letterbox_scale', 1.0)
        letterbox_pad = getattr(self.infer_engine, 'letterbox_pad', (0, 0))
        dw, dh = letterbox_pad

        keypoints_xy = []
        keypoints_conf = []

        for i in range(min(detected_keypoints, NUM_KEYPOINTS)):
            idx = 5 + i * 3
            if idx + 2 < len(best_detection):
                kp_x = best_detection[idx]
                kp_y = best_detection[idx + 1]
                kp_conf = best_detection[idx + 2]

                kp_x_unpad = kp_x - dw
                kp_y_unpad = kp_y - dh

                kp_x_scaled = kp_x_unpad / letterbox_scale
                kp_y_scaled = kp_y_unpad / letterbox_scale
            else:
                kp_x_scaled, kp_y_scaled, kp_conf = 0.0, 0.0, 0.0

            keypoints_xy.append([kp_x_scaled, kp_y_scaled])
            keypoints_conf.append(kp_conf)

        while len(keypoints_xy) < NUM_KEYPOINTS:
            keypoints_xy.append([0.0, 0.0])
            keypoints_conf.append(0.0)

        keypoints_xy = np.array([keypoints_xy], dtype=np.float32)
        keypoints_conf = np.array([keypoints_conf], dtype=np.float32)

        return keypoints_xy, keypoints_conf


class DeepStreamSegmentationModel:
    
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.model_name = "Segmentation Model (TensorRT)"
        
        import sys
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from config import PROJECT_DIR
        
        self.engine_path = os.path.join(PROJECT_DIR, "models/yolov8m-seg.engine")
        self.confidence_threshold = 0.3
        self.default_classes = None
        
        self._load_config(PROJECT_DIR)
        
        if not os.path.exists(self.engine_path):
            print(f"Engine file not found: {self.engine_path}")
            print(f"Please run convert_to_engine.py script")
            self.infer_engine = None
        else:
            self.infer_engine = create_tensorrt_inference(self.engine_path)
            if self.infer_engine:
                print(f"{self.model_name} TensorRT engine loaded")
    
    def _load_config(self, project_dir: str):
        try:
            current_section = None
            with open(self.config_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('['):
                        current_section = line.strip('[]')
                    elif '=' in line and not line.startswith('#'):
                        key, value = line.split('=', 1)
                        key = key.strip()
                        value = value.strip()
                        
                        if current_section == 'model':
                            if key == 'engine-file':
                                self.engine_path = os.path.join(project_dir, value)
                            elif key == 'confidence-threshold':
                                self.confidence_threshold = float(value)
                        elif current_section == 'processing':
                            if key == 'classes':
                                self.default_classes = [int(c.strip()) for c in value.split(',')]
        except Exception as e:
            print(f"Segmentation config load error: {e}, using defaults")
    
    def __call__(
        self,
        frame: np.ndarray,
        conf: float = None,
        classes: Optional[List[int]] = None,
        verbose: bool = False
    ) -> List:
        if conf is None:
            conf = self.confidence_threshold
        if classes is None:
            classes = self.default_classes
        if self.infer_engine is None:
            boxes = np.zeros((0, 4), dtype=np.float32)
            masks = np.zeros((0, 640, 640), dtype=np.float32)
            confidences = np.zeros((0,), dtype=np.float32)
            result = SegmentationResult(boxes, masks, confidences)
            return [result]

        outputs = self.infer_engine.infer(frame)

        boxes, masks, confidences = self._parse_seg_output(
            outputs,
            conf,
            classes,
            frame.shape
        )

        result = SegmentationResult(boxes, masks, confidences)
        return [result]
    
    def _parse_seg_output(
        self,
        outputs: List[np.ndarray],
        conf_threshold: float,
        classes: Optional[List[int]],
        frame_shape: Tuple[int, int, int]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        MODEL_INPUT_SIZE = 640  # TensorRT model input size

        if len(outputs) < 2:
            return (
                np.zeros((0, 4), dtype=np.float32),
                np.zeros((0, 640, 640), dtype=np.float32),
                np.zeros((0,), dtype=np.float32)
            )

        output0 = outputs[0]
        output1 = outputs[1]

        if len(output0.shape) == 3:
            output0 = output0[0].T
        boxes_xywh = output0[:, :4]
        class_probs = output0[:, 4:84]
        mask_coeffs = output0[:, 84:]

        class_ids = np.argmax(class_probs, axis=1)
        confidences = np.max(class_probs, axis=1)

        conf_mask = confidences > conf_threshold

        if classes is not None:
            class_mask = np.isin(class_ids, classes)
            conf_mask = conf_mask & class_mask

        boxes_xywh = boxes_xywh[conf_mask]
        confidences = confidences[conf_mask]
        mask_coeffs = mask_coeffs[conf_mask]
        class_ids = class_ids[conf_mask]

        if len(boxes_xywh) == 0:
            return (
                np.zeros((0, 4), dtype=np.float32),
                np.zeros((0, 640, 640), dtype=np.float32),
                np.zeros((0,), dtype=np.float32)
            )

        letterbox_scale = getattr(self.infer_engine, 'letterbox_scale', 1.0)
        letterbox_pad = getattr(self.infer_engine, 'letterbox_pad', (0, 0))
        dw, dh = letterbox_pad

        boxes_xyxy = np.zeros_like(boxes_xywh)

        for i in range(len(boxes_xywh)):
            x_center, y_center, width, height = boxes_xywh[i]

            x1 = x_center - width / 2
            y1 = y_center - height / 2
            x2 = x_center + width / 2
            y2 = y_center + height / 2

            x1_unpad = x1 - dw
            y1_unpad = y1 - dh
            x2_unpad = x2 - dw
            y2_unpad = y2 - dh

            boxes_xyxy[i, 0] = x1_unpad / letterbox_scale
            boxes_xyxy[i, 1] = y1_unpad / letterbox_scale
            boxes_xyxy[i, 2] = x2_unpad / letterbox_scale
            boxes_xyxy[i, 3] = y2_unpad / letterbox_scale

        keep_indices = nms_boxes(boxes_xyxy, confidences, iou_threshold=0.7)

        if len(keep_indices) == 0:
            return (
                np.zeros((0, 4), dtype=np.float32),
                np.zeros((0, 640, 640), dtype=np.float32),
                np.zeros((0,), dtype=np.float32)
            )

        boxes_xyxy = boxes_xyxy[keep_indices]
        confidences = confidences[keep_indices]
        mask_coeffs = mask_coeffs[keep_indices]
        class_ids = class_ids[keep_indices]

        masks = self._generate_masks(output1, mask_coeffs, boxes_xyxy, frame_shape)

        return boxes_xyxy, masks, confidences

    def _generate_masks(
        self,
        proto_masks: np.ndarray,
        mask_coeffs: np.ndarray,
        boxes: np.ndarray,
        frame_shape: Tuple[int, int, int]
    ) -> np.ndarray:
        if len(proto_masks.shape) == 4:
            proto_masks = proto_masks[0]

        num_masks, mh, mw = proto_masks.shape
        num_detections = len(mask_coeffs)

        if num_detections == 0:
            return np.zeros((0, frame_shape[0], frame_shape[1]), dtype=np.float32)

        proto_flat = proto_masks.reshape(num_masks, -1)
        masks_flat = mask_coeffs @ proto_flat

        masks_flat = 1 / (1 + np.exp(-masks_flat))

        masks = masks_flat.reshape(num_detections, mh, mw)

        letterbox_scale = getattr(self.infer_engine, 'letterbox_scale', 1.0)
        letterbox_pad = getattr(self.infer_engine, 'letterbox_pad', (0, 0))
        dw, dh = letterbox_pad

        masks_resized = np.zeros((num_detections, 640, 640), dtype=np.float32)
        for i in range(num_detections):
            masks_resized[i] = cv2.resize(masks[i], (640, 640), interpolation=cv2.INTER_LINEAR)

        orig_h, orig_w = frame_shape[:2]
        final_masks = np.zeros((num_detections, orig_h, orig_w), dtype=np.float32)

        for i in range(num_detections):
            x1_crop = int(dw)
            y1_crop = int(dh)
            x2_crop = int(640 - dw)
            y2_crop = int(640 - dh)

            mask_cropped = masks_resized[i, y1_crop:y2_crop, x1_crop:x2_crop]

            if mask_cropped.size > 0:
                final_masks[i] = cv2.resize(mask_cropped, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

            x1, y1, x2, y2 = boxes[i].astype(int)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(orig_w, x2), min(orig_h, y2)

            box_mask = np.zeros((orig_h, orig_w), dtype=np.float32)
            box_mask[y1:y2, x1:x2] = 1.0
            final_masks[i] *= box_mask

        final_masks = (final_masks > 0.5).astype(np.float32)

        return final_masks


class PoseResult:
    
    def __init__(self, keypoints_xy: np.ndarray, keypoints_conf: np.ndarray):
        self.keypoints = KeypointsWrapper(keypoints_xy, keypoints_conf)


class KeypointsWrapper:

    def __init__(self, xy: np.ndarray, conf: np.ndarray):
        self._xy = xy
        self._conf = conf

    @property
    def xy(self):
        return MockTensor(self._xy)

    @property
    def conf(self):
        return MockTensor(self._conf)


class SegmentationResult:
    
    def __init__(self, boxes: np.ndarray, masks: np.ndarray, confidences: np.ndarray):
        self.boxes = BoxesWrapper(boxes, confidences)
        self.masks = MasksWrapper(masks) if len(masks) > 0 else None


class BoxesWrapper:
    
    def __init__(self, boxes: np.ndarray, confidences: np.ndarray):
        self._boxes = boxes
        self._confidences = confidences
    
    @property
    def xyxy(self):
        return MockTensor(self._boxes)
    
    @property
    def conf(self):
        return MockTensor(self._confidences)


class MasksWrapper:
    
    def __init__(self, masks: np.ndarray):
        self._masks = masks
    
    @property
    def data(self):
        return MockTensor(self._masks)


class MockTensor:

    def __init__(self, data: np.ndarray):
        self._data = data

    def __len__(self):
        return len(self._data)

    def __getitem__(self, key):
        result = self._data[key]
        if isinstance(result, np.ndarray):
            return MockTensor(result)
        return result

    def cpu(self):
        return self

    def numpy(self):
        return self._data

def nms_boxes(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.7) -> np.ndarray:
    if len(boxes) == 0:
        return np.array([], dtype=np.int32)

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1) * (y2 - y1)

    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        intersection = w * h

        iou = intersection / (areas[i] + areas[order[1:]] - intersection)

        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]

    return np.array(keep, dtype=np.int32)

def create_deepstream_pose_model(config_path: str) -> DeepStreamPoseModel:
    return DeepStreamPoseModel(config_path)


def create_deepstream_seg_model(config_path: str) -> DeepStreamSegmentationModel:
    return DeepStreamSegmentationModel(config_path)

