import cv2
import numpy as np


def filter_detections_by_roi(boxes, masks, hull):
    roi_indices = []
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        
        foot_x = (x1 + x2) / 2
        foot_y = y2
        
        foot_in = cv2.pointPolygonTest(hull, (foot_x, foot_y), False) >= 0
        
        if foot_in:
            roi_indices.append(i)

    return roi_indices


def draw_player_mask(frame, mask, w, h):
    mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

    colored_mask = np.zeros_like(frame)
    colored_mask[mask_resized > 0.5] = (255, 100, 0)
    cv2.addWeighted(frame, 1, colored_mask, 0.5, 0, dst=frame)


def draw_player_box(frame, box, player_id):
    x1, y1, x2, y2 = box
    
    foot_x = (x1 + x2) / 2
    foot_y = y2
    
    text = f"ID:{player_id}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    thickness = 2
    
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_x = int(foot_x - text_size[0] / 2)
    text_y = int(foot_y - 15)
    
    padding = 5
    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (text_x - padding, text_y - text_size[1] - padding),
        (text_x + text_size[0] + padding, text_y + padding),
        (0, 0, 0),
        -1
    )
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    
    cv2.putText(
        frame,
        text,
        (text_x, text_y),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
    )


def process_players(
    frame,
    seg_results,
    hull,
    tracker,
    player_tracker,
    H,
    frame_idx,
    w,
    h,
    image_to_court,
    COURT_LENGTH,
    COURT_WIDTH,
):
    player_foot_positions = []
    
    if seg_results.boxes is None or seg_results.masks is None:
        return player_foot_positions

    boxes_orig = seg_results.boxes.xyxy.cpu().numpy()
    masks_orig = seg_results.masks.data.cpu().numpy()
    confidences_orig = seg_results.boxes.conf.cpu().numpy()

    roi_indices = filter_detections_by_roi(boxes_orig, masks_orig, hull)
    
    total_detections = len(boxes_orig)
    roi_detections = len(roi_indices)
    filtered_out = total_detections - roi_detections
    
    if frame_idx % 30 == 0:
        print(f"Detection Frame {frame_idx}: {total_detections} total -> {roi_detections} in ROI (Filtered: {filtered_out})")

    if len(roi_indices) == 0:
        return player_foot_positions

    boxes_roi = boxes_orig[roi_indices]
    masks_roi = masks_orig[roi_indices]
    confidences_roi = confidences_orig[roi_indices]
    
    if tracker is None:
        for i, box in enumerate(boxes_roi):
            x1, y1, x2, y2 = box
            
            foot_x = (x1 + x2) / 2
            foot_y = y2
            
            player_foot_positions.append((foot_x, foot_y))

            draw_player_mask(frame, masks_roi[i], w, h)

            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 100, 0), 2)
            cv2.putText(
                frame,
                "Player",
                (int(x1), int(y1) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )
        
        return player_foot_positions
    
    detections = []
    for i, box in enumerate(boxes_roi):
        x1, y1, x2, y2 = box
        ltwh = [x1, y1, x2 - x1, y2 - y1]
        confidence = confidences_roi[i]
        detection_class = 0

        detections.append((ltwh, confidence, detection_class))

    tracks = tracker.update_tracks(detections, frame=frame)
    
    confirmed_tracks = [t for t in tracks if t.is_confirmed() and t.time_since_update == 0]
    
    detections_before = len(detections)
    detections_after = len(confirmed_tracks)
    loss = detections_before - detections_after
    if frame_idx % 30 == 0:
        status = "Warning" if loss >= 2 else "OK"
        print(f"{status} - Tracking Frame {frame_idx}: {detections_before} detection -> {detections_after} tracked (Loss: {loss})")

    mask_idx = 0
    for track in tracks:
        if not track.is_confirmed():
            continue
        
        if track.time_since_update > 0:
            continue
            
        track_id = track.track_id
        ltwh = track.to_ltwh()
        
        x1, y1, w_box, h_box = ltwh
        x2 = x1 + w_box
        y2 = y1 + h_box
        box = [x1, y1, x2, y2]
        
        foot_x = (x1 + x2) / 2
        foot_y = y2
        
        foot_in_roi = cv2.pointPolygonTest(hull, (foot_x, foot_y), False) >= 0
        if not foot_in_roi:
            continue
        
        player_foot_positions.append((foot_x, foot_y))

        if mask_idx < len(masks_roi):
            draw_player_mask(frame, masks_roi[mask_idx], w, h)
            mask_idx += 1

        draw_player_box(frame, box, track_id)

        if H is not None and player_tracker is not None:
            foot_pt = np.array([[foot_x, foot_y]], dtype=np.float32)
            court_pos = image_to_court(H, foot_pt)[0]

            if 0 <= court_pos[0] <= COURT_LENGTH and 0 <= court_pos[1] <= COURT_WIDTH:
                player_tracker.update(track_id, court_pos, frame_idx)

    return player_foot_positions
