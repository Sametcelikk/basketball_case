import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)

print("DeepStream backend (TensorRT)")

from deepstream_inference import (
    create_deepstream_pose_model,
    create_deepstream_seg_model
)

POSE_CONFIG = os.path.join(PROJECT_DIR, "deepstream_configs/config_pose.txt")
SEG_CONFIG = os.path.join(PROJECT_DIR, "deepstream_configs/config_segmentation.txt")

seg_model = create_deepstream_seg_model(SEG_CONFIG)
pose_model = create_deepstream_pose_model(POSE_CONFIG)

print("DeepStream Segmentation model loaded")
print("DeepStream Pose model loaded")
print("TensorRT engine active")

input_video = os.path.join(PROJECT_DIR, "static/input.mp4")
