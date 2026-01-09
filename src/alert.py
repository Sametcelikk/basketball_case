import cv2
import numpy as np
from court_utils import COURT_KEYPOINTS, project_points


class PaintAreaAlert:
    
    def __init__(self, threshold=2, warmup_frames=15):
        self.threshold = threshold
        self.warmup_frames = warmup_frames
        self.frame_count = 0
        
        self.left_paint_keypoints = [2, 9, 11, 3]
        self.right_paint_keypoints = [21, 29, 30, 23]
        
        self.left_alert_active = False
        self.right_alert_active = False
        self.alert_cooldown = 0
        
    def get_paint_polygon(self, H, keypoint_ids, last_kpts=None, last_conf=None):
        if H is None:
            return None
        
        critical_detected = 0
        for kpt_id in keypoint_ids:
            if last_kpts is not None and last_conf is not None:
                if kpt_id < len(last_conf) and last_conf[kpt_id] > 0.5:
                    critical_detected += 1
        
        if critical_detected < 3:
            return None
        
        image_points = []
        
        for kpt_id in keypoint_ids:
            if last_kpts is not None and last_conf is not None:
                if kpt_id < len(last_conf) and last_conf[kpt_id] > 0.5:
                    image_points.append(last_kpts[kpt_id])
                    continue
            
            court_point = COURT_KEYPOINTS[kpt_id:kpt_id+1]
            projected = project_points(H, court_point)[0]
            image_points.append(projected)
        
        polygon = np.array(image_points, dtype=np.int32)
        
        return polygon
    
    def count_players_in_area(self, player_positions, hull):
        if hull is None or len(player_positions) == 0:
            return 0
            
        count = 0
        for pos in player_positions:
            x, y = pos
            if cv2.pointPolygonTest(hull, (float(x), float(y)), False) >= 0:
                count += 1
                
        return count
    
    def draw_paint_area(self, frame, hull, is_alert, side_name, player_count):
        if hull is None:
            return
            
        overlay = frame.copy()
        
        if is_alert:
            color = (0, 0, 255)
            alpha = 0.4
        else:
            color = (0, 200, 200)
            alpha = 0.15
            
        cv2.fillPoly(overlay, [hull], color)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        border_color = (0, 0, 255) if is_alert else (0, 200, 200)
        cv2.polylines(frame, [hull], True, border_color, 2)
        
        if len(hull) > 0:
            M = cv2.moments(hull)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                
                count_text = str(player_count)
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 2.0
                thickness = 4
                
                text_size = cv2.getTextSize(count_text, font, font_scale, thickness)[0]
                
                padding = 15
                box_x1 = cx - text_size[0] // 2 - padding
                box_y1 = cy - text_size[1] // 2 - padding
                box_x2 = cx + text_size[0] // 2 + padding
                box_y2 = cy + text_size[1] // 2 + padding
                
                overlay2 = frame.copy()
                cv2.rectangle(overlay2, (box_x1, box_y1), (box_x2, box_y2), (0, 0, 0), -1)
                cv2.addWeighted(overlay2, 0.7, frame, 0.3, 0, frame)
                
                cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), border_color, 3)
                
                text_x = cx - text_size[0] // 2
                text_y = cy + text_size[1] // 2
                cv2.putText(
                    frame,
                    count_text,
                    (text_x, text_y),
                    font,
                    font_scale,
                    (255, 255, 255),
                    thickness,
                )
        
    def draw_player_count_info(self, frame, w, h, left_count, right_count, left_alert, right_alert):
        if left_alert or right_alert:
            self.draw_big_alert(frame, w, h)
    
    def check_and_draw(self, frame, H, player_positions, w, h, last_kpts=None, last_conf=None, h_history_len=0):
        self.frame_count += 1
        
        if H is None:
            return False, False
        
        min_h_history = 3
        if self.frame_count <= self.warmup_frames or h_history_len < min_h_history:
            left_hull = self.get_paint_polygon(H, self.left_paint_keypoints, last_kpts, last_conf)
            right_hull = self.get_paint_polygon(H, self.right_paint_keypoints, last_kpts, last_conf)
            
            if left_hull is not None:
                self.draw_paint_area(frame, left_hull, False, "LEFT", 0)
            
            if right_hull is not None:
                self.draw_paint_area(frame, right_hull, False, "RIGHT", 0)
            
            return False, False
            
        left_hull = self.get_paint_polygon(H, self.left_paint_keypoints, last_kpts, last_conf)
        
        if left_hull is None:
            left_count = 0
            left_alert = False
        else:
            left_count = self.count_players_in_area(player_positions, left_hull)
            left_alert = left_count >= self.threshold
        
        right_hull = self.get_paint_polygon(H, self.right_paint_keypoints, last_kpts, last_conf)
        
        if right_hull is None:
            right_count = 0
            right_alert = False
        else:
            right_count = self.count_players_in_area(player_positions, right_hull)
            right_alert = right_count >= self.threshold
        
        self.draw_paint_area(frame, left_hull, left_alert, "LEFT", left_count)
        self.draw_paint_area(frame, right_hull, right_alert, "RIGHT", right_count)
        
        self.draw_player_count_info(frame, w, h, left_count, right_count, left_alert, right_alert)
            
        return left_alert, right_alert
    
    def draw_big_alert(self, frame, w, h):
        base_scale = min(w / 1280.0, h / 720.0)
        font_scale = 0.45 * base_scale
        thickness = max(1, int(1.5 * base_scale))
        padding = int(8 * base_scale)
        
        title = "PLAYER DENSITY ALERT"
        font = cv2.FONT_HERSHEY_SIMPLEX
        title_size = cv2.getTextSize(title, font, font_scale, thickness)[0]
        
        title_x = 15
        title_y = 15 + title_size[1]
        
        overlay = frame.copy()
        cv2.rectangle(overlay, (title_x - padding, title_y - title_size[1] - 5), 
                     (title_x + title_size[0] + padding, title_y + 5), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        cv2.rectangle(frame, (title_x - padding, title_y - title_size[1] - 5), 
                     (title_x + title_size[0] + padding, title_y + 5), (0, 0, 255), 2)
        
        cv2.putText(
            frame,
            title,
            (title_x, title_y),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
        )

