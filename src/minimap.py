import cv2
import numpy as np
import os

from court_utils import COURT_LENGTH, COURT_WIDTH

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
minimap_img = cv2.imread(os.path.join(PROJECT_DIR, "static", "court.jpeg"))
MINIMAP_H, MINIMAP_W = minimap_img.shape[:2]


def court_to_minimap(court_pts):
    if court_pts.ndim == 1:
        court_pts = court_pts.reshape(1, -1)
    minimap_x = (court_pts[:, 0] / COURT_LENGTH) * MINIMAP_W
    minimap_y = (court_pts[:, 1] / COURT_WIDTH) * MINIMAP_H
    return np.column_stack([minimap_x, minimap_y])


def draw_players_on_minimap(current_minimap, active_players):
    for player_id, court_pos in active_players.items():
        minimap_pos = court_to_minimap(court_pos.reshape(1, -1))[0]
        mx, my = int(minimap_pos[0]), int(minimap_pos[1])

        if 0 <= mx < MINIMAP_W and 0 <= my < MINIMAP_H:
            cv2.circle(current_minimap, (mx, my), 18, (0, 0, 255), -1)
            cv2.circle(current_minimap, (mx, my), 18, (255, 255, 255), 3)
            
            text = str(player_id)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            
            text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
            text_x = mx - text_size[0] // 2
            text_y = my + text_size[1] // 2
            
            cv2.putText(
                current_minimap,
                text,
                (text_x, text_y),
                font,
                font_scale,
                (255, 255, 255),
                thickness,
            )


def overlay_minimap_on_frame(frame, current_minimap, h, w=None, minimap_scale=None):
    if minimap_scale is None and w is not None:
        base_width = 1280.0
        minimap_scale = 0.35 * (w / base_width)
        minimap_scale = max(0.20, min(minimap_scale, 0.6))
    elif minimap_scale is None:
        minimap_scale = 0.5
    
    minimap_resized = cv2.resize(
        current_minimap,
        (int(MINIMAP_W * minimap_scale), int(MINIMAP_H * minimap_scale)),
    )
    mh, mw = minimap_resized.shape[:2]

    x_offset = 15
    y_offset = h - mh - 15

    roi = frame[y_offset : y_offset + mh, x_offset : x_offset + mw]
    blended = cv2.addWeighted(roi, 0.2, minimap_resized, 0.8, 0)
    frame[y_offset : y_offset + mh, x_offset : x_offset + mw] = blended
