import cv2
import os

from config import PROJECT_DIR, input_video


def create_video_capture():
    cap = cv2.VideoCapture(input_video)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return cap, fps, w, h


def create_video_writer(fps, w, h, output_name="output.mp4"):
    output_path = os.path.join(PROJECT_DIR, output_name)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    return cv2.VideoWriter(output_path, fourcc, fps, (w, h))
