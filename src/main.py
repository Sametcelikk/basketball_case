import numpy as np
import cv2
import os
from collections import deque
from typing import Callable, Optional

from config import seg_model, pose_model, PROJECT_DIR
from court_utils import (
    COURT_LENGTH,
    COURT_WIDTH,
    find_homography,
    image_to_court,
    process_keypoints,
    smooth_keypoints,
    create_roi,
    smooth_roi,
    draw_roi,
    draw_keypoints,
)
from tracker import create_tracker, PlayerTracker
from segmentation import process_players
from minimap import minimap_img, draw_players_on_minimap, overlay_minimap_on_frame
from video import create_video_capture, create_video_writer
from alert import PaintAreaAlert


def process_video(
    input_path: str,
    output_path: str,
    progress_callback: Optional[Callable[[int, int], None]] = None
) -> str:
    cap = cv2.VideoCapture(input_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    frame_idx = 0
    last_kpts = None
    last_conf = None
    smoothed_pts = None
    H_history = deque(maxlen=5)
    roi_state = None
    
    stable_H = None
    stable_roi_hull = None
    last_roi_update_frame = -999
    ROI_UPDATE_INTERVAL = 5
    ROI_CHANGE_THRESHOLD = 0.20

    cached_seg_results = None
    SEG_SKIP_FRAMES = 2

    tracker_config = os.path.join(PROJECT_DIR, "deepstream_configs/config_tracker.txt")
    tracker = create_tracker(tracker_config)
    player_tracker = PlayerTracker(config_path=tracker_config)

    print(f"Processing video: {input_path}")
    print(f"Total frames: {total_frames}, FPS: {fps}, Size: {w}x{h}")
    print("Tracker initialized")

    paint_alert = PaintAreaAlert(threshold=2)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % 3 == 0:
            results = pose_model(frame, verbose=False)[0]
            if results.keypoints is not None and len(results.keypoints.xy) > 0:
                last_kpts = results.keypoints.xy[0].cpu().numpy()
                last_conf = results.keypoints.conf[0].cpu().numpy()

        if last_kpts is not None:
            H = find_homography(last_kpts, last_conf)
            if H is not None:
                H_history.append(H)
                if len(H_history) > 0:
                    H = np.mean(H_history, axis=0)

            all_pts, _ = process_keypoints(last_kpts, last_conf, H, w, h)
            smoothed_pts = smooth_keypoints(all_pts, smoothed_pts)
            all_pts = smoothed_pts

            current_roi = create_roi(all_pts, w, h)
            hull, roi_state = smooth_roi(current_roi, roi_state, alpha=0.3, max_frames_without_roi=10)
            
            should_update_roi = (frame_idx - last_roi_update_frame) >= ROI_UPDATE_INTERVAL
            
            if should_update_roi and hull is not None:
                roi_changed_significantly = False
                avg_diff = 0.0
                max_diff = 0.0
                area_change_ratio = 0.0
                
                if stable_roi_hull is None:
                    roi_changed_significantly = True
                else:
                    if len(hull) == len(stable_roi_hull):
                        hull_diff = np.linalg.norm(hull.astype(float) - stable_roi_hull.astype(float), axis=1)
                        avg_diff = np.mean(hull_diff)
                        max_diff = np.max(hull_diff)
                        
                        if avg_diff > 20 or max_diff > 60:
                            roi_changed_significantly = True
                    else:
                        area_old = cv2.contourArea(stable_roi_hull)
                        area_new = cv2.contourArea(hull)
                        if area_old > 0:
                            area_change_ratio = abs(area_new - area_old) / area_old
                            if area_change_ratio > ROI_CHANGE_THRESHOLD:
                                roi_changed_significantly = True
                        else:
                            roi_changed_significantly = True
                
                if roi_changed_significantly:
                    stable_roi_hull = hull.copy()
                    stable_H = H.copy() if H is not None else stable_H
                    last_roi_update_frame = frame_idx
            
            if stable_roi_hull is None and hull is not None:
                stable_roi_hull = hull.copy()
                stable_H = H.copy() if H is not None else None
                last_roi_update_frame = frame_idx
            
            if hull is not None:
                draw_roi(frame, hull)
                current_minimap = minimap_img.copy()

                if frame_idx % SEG_SKIP_FRAMES == 0 or cached_seg_results is None:
                    cached_seg_results = seg_model(frame, verbose=False)[0]

                player_foot_positions = process_players(
                    frame,
                    cached_seg_results,
                    hull,
                    tracker,
                    player_tracker,
                    stable_H if stable_H is not None else H,
                    frame_idx,
                    w,
                    h,
                    image_to_court,
                    COURT_LENGTH,
                    COURT_WIDTH,
                )

                paint_alert.check_and_draw(frame, H, player_foot_positions, w, h, last_kpts, last_conf, len(H_history))

                active_players = player_tracker.get_all_active_players(frame_idx)
                
                draw_players_on_minimap(current_minimap, active_players)
                overlay_minimap_on_frame(frame, current_minimap, h, w)

            draw_keypoints(frame, all_pts, last_conf, w, h)

        out.write(frame)
        frame_idx += 1

        if progress_callback and frame_idx % 30 == 0:
            progress_callback(frame_idx, total_frames)

    cap.release()
    out.release()
    
    print(f"Processing completed: {output_path}")

    return output_path


def main():
    input_path = os.path.join(PROJECT_DIR, "static/input.mp4")
    output_path = os.path.join(PROJECT_DIR, "output.mp4")

    process_video(input_path, output_path)
    print("Done")


if __name__ == "__main__":
    main()
