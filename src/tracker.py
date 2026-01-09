
import numpy as np
import cv2
from typing import List, Tuple, Optional
import os

class Tracker:

    def __init__(self, config_path: str):
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")

        self.config_path = config_path
        self.tracker_width = 640
        self.tracker_height = 640
        
        self._load_config()

        self.tracks = {}
        self.frame_count = 0
        self.next_track_id = 1
    
    def _load_config(self):
        try:
            with open(self.config_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if '=' in line and not line.startswith('['):
                        key, value = line.split('=', 1)
                        key = key.strip()
                        value = value.strip()
                        
                        if key == 'tracker-width':
                            self.tracker_width = int(value)
                        elif key == 'tracker-height':
                            self.tracker_height = int(value)
        except Exception as e:
            print(f"Config load error: {e}, using defaults")

    def update_tracks(
        self,
        detections: List[Tuple],
        frame: Optional[np.ndarray] = None
    ) -> List['Track']:
        if frame is None:
            raise ValueError("Frame required for appearance features.")

        self.frame_count += 1
        tracked_objects = self._track_detections(detections, frame)

        return tracked_objects

    def _track_detections(
        self,
        detections: List[Tuple],
        frame: np.ndarray
    ) -> List['Track']:
        object_metas = []
        
        for idx, (ltwh, conf, cls) in enumerate(detections):
            x1, y1, w, h = ltwh
            x2, y2 = x1 + w, y1 + h
            
            x1_int, y1_int = max(0, int(x1)), max(0, int(y1))
            x2_int, y2_int = min(frame.shape[1], int(x2)), min(frame.shape[0], int(y2))
            crop = frame[y1_int:y2_int, x1_int:x2_int]
            
            obj_meta = {
                'bbox': [x1, y1, x2, y2],
                'confidence': conf,
                'class_id': cls,
                'crop': crop,
                'object_id': -1,
            }
            object_metas.append(obj_meta)

        tracked_metas = self._process_tracking(object_metas, frame)

        tracks = []
        for meta in tracked_metas:
            if meta['object_id'] in self.tracks:
                track = self.tracks[meta['object_id']]
                track.update(
                    np.array(meta['bbox']),
                    meta.get('feature', np.zeros(128)),
                    meta['confidence']
                )
            else:
                track = Track(
                    track_id=meta['object_id'],
                    box=np.array(meta['bbox']),
                    feature=meta.get('feature', np.zeros(128)),
                    confidence=meta['confidence'],
                    n_init=8
                )
                self.tracks[meta['object_id']] = track
            
            tracks.append(track)

        self._cleanup_old_tracks(max_age=90)

        return [t for t in tracks if t.is_confirmed() and t.time_since_update == 0]

    def _process_tracking(
        self,
        object_metas: List[dict],
        frame: np.ndarray
    ) -> List[dict]:
        for meta in object_metas:
            meta['feature'] = self._extract_reid_feature(meta['crop'])

        det_boxes = np.array([m['bbox'] for m in object_metas])
        det_features = np.array([m['feature'] for m in object_metas])
        det_confs = np.array([m['confidence'] for m in object_metas])

        track_ids = list(self.tracks.keys())
        track_boxes = np.array([self.tracks[tid].box for tid in track_ids]) if track_ids else np.zeros((0, 4))
        track_features = np.array([self.tracks[tid].feature for tid in track_ids]) if track_ids else np.zeros((0, 128))

        matched_indices, unmatched_dets, unmatched_tracks = self._hungarian_matching(
            det_boxes, track_boxes, det_features, track_features, track_ids
        )

        tracked_metas = []

        for det_idx, track_idx in matched_indices:
            track_id = track_ids[track_idx]
            meta = object_metas[det_idx].copy()
            meta['object_id'] = track_id
            tracked_metas.append(meta)

        for det_idx in unmatched_dets:
            meta = object_metas[det_idx].copy()
            meta['object_id'] = self._get_next_track_id()
            tracked_metas.append(meta)

        for track_idx in unmatched_tracks:
            track_id = track_ids[track_idx]
            if track_id in self.tracks:
                self.tracks[track_id].mark_missed()

        return tracked_metas

    def _extract_reid_feature(self, crop: np.ndarray) -> np.ndarray:
        if crop.size == 0:
            return np.zeros(128, dtype=np.float32)

        crop_resized = cv2.resize(crop, (64, 128))

        hsv = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2HSV)
        hist_h = cv2.calcHist([hsv], [0], None, [16], [0, 180]).flatten()
        hist_s = cv2.calcHist([hsv], [1], None, [16], [0, 256]).flatten()
        hist_v = cv2.calcHist([hsv], [2], None, [16], [0, 256]).flatten()
        
        hist_h = hist_h / (hist_h.sum() + 1e-6)
        hist_s = hist_s / (hist_s.sum() + 1e-6)
        hist_v = hist_v / (hist_v.sum() + 1e-6)
        
        color_feature = np.concatenate([hist_h, hist_s, hist_v])

        gray = cv2.cvtColor(crop_resized, cv2.COLOR_BGR2GRAY)
        
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        
        magnitude = np.sqrt(gx**2 + gy**2)
        direction = np.arctan2(gy, gx) * 180 / np.pi
        direction[direction < 0] += 360
        
        hog_hist, _ = np.histogram(direction, bins=9, range=(0, 360), weights=magnitude)
        hog_hist = hog_hist / (hog_hist.sum() + 1e-6)
        
        h, w = gray.shape
        cell_h, cell_w = h // 2, w // 2
        hog_features = []
        
        for i in range(2):
            for j in range(2):
                cell = gray[i*cell_h:(i+1)*cell_h, j*cell_w:(j+1)*cell_w]
                cell_gx = cv2.Sobel(cell, cv2.CV_32F, 1, 0, ksize=3)
                cell_gy = cv2.Sobel(cell, cv2.CV_32F, 0, 1, ksize=3)
                cell_mag = np.sqrt(cell_gx**2 + cell_gy**2)
                cell_dir = np.arctan2(cell_gy, cell_gx) * 180 / np.pi
                cell_dir[cell_dir < 0] += 360
                cell_hist, _ = np.histogram(cell_dir, bins=9, range=(0, 360), weights=cell_mag)
                cell_hist = cell_hist / (cell_hist.sum() + 1e-6)
                hog_features.append(cell_hist)
        
        hog_feature = np.concatenate(hog_features)

        upper = crop_resized[:64, :]
        upper_hsv = cv2.cvtColor(upper, cv2.COLOR_BGR2HSV)
        
        upper_hist_h = cv2.calcHist([upper_hsv], [0], None, [12], [0, 180]).flatten()
        upper_hist_s = cv2.calcHist([upper_hsv], [1], None, [12], [0, 256]).flatten()
        upper_hist_h = upper_hist_h / (upper_hist_h.sum() + 1e-6)
        upper_hist_s = upper_hist_s / (upper_hist_s.sum() + 1e-6)
        
        lower = crop_resized[64:, :]
        lower_hsv = cv2.cvtColor(lower, cv2.COLOR_BGR2HSV)
        lower_hist_h = cv2.calcHist([lower_hsv], [0], None, [6], [0, 180]).flatten()
        lower_hist_s = cv2.calcHist([lower_hsv], [1], None, [6], [0, 256]).flatten()
        lower_hist_h = lower_hist_h / (lower_hist_h.sum() + 1e-6)
        lower_hist_s = lower_hist_s / (lower_hist_s.sum() + 1e-6)
        
        spatial_feature = np.concatenate([upper_hist_h, upper_hist_s, lower_hist_h, lower_hist_s])
        
        aspect_ratio = crop.shape[1] / (crop.shape[0] + 1e-6)
        size_feature = np.array([
            aspect_ratio,
            crop.shape[0] / 200.0,
            crop.shape[1] / 100.0,
        ])
        
        upper_mean_h = np.mean(upper_hsv[:, :, 0])
        upper_mean_s = np.mean(upper_hsv[:, :, 1])
        upper_mean_v = np.mean(upper_hsv[:, :, 2])
        
        dominant_color = np.array([upper_mean_h / 180.0, upper_mean_s / 255.0, upper_mean_v / 255.0])
        
        size_and_color = np.concatenate([size_feature, dominant_color])
        size_and_color = np.pad(size_and_color, (0, 8 - len(size_and_color)), mode='constant')
        
        spatial_feature = np.concatenate([spatial_feature, size_and_color])

        feature = np.concatenate([color_feature, hog_feature, spatial_feature])
        feature = feature / (np.linalg.norm(feature) + 1e-6)
        
        return feature.astype(np.float32)

    def _hungarian_matching(
        self,
        det_boxes: np.ndarray,
        track_boxes: np.ndarray,
        det_features: np.ndarray,
        track_features: np.ndarray,
        track_ids: List[int]
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        if len(det_boxes) == 0 or len(track_boxes) == 0:
            return [], list(range(len(det_boxes))), list(range(len(track_boxes)))

        cost_matrix = self._compute_cost_matrix(
            det_boxes, track_boxes, det_features, track_features
        )

        try:
            from scipy.optimize import linear_sum_assignment
            row_indices, col_indices = linear_sum_assignment(cost_matrix)
        except ImportError:
            return self._greedy_matching(cost_matrix)

        matched_indices = []
        unmatched_dets = list(range(len(det_boxes)))
        unmatched_tracks = list(range(len(track_boxes)))

        threshold = 0.5

        for det_idx, track_idx in zip(row_indices, col_indices):
            track_id = track_ids[track_idx]
            track = self.tracks.get(track_id)
            
            if track and track.is_confirmed():
                current_threshold = 0.5
            else:
                current_threshold = 0.3
            
            if cost_matrix[det_idx, track_idx] < current_threshold:
                matched_indices.append((det_idx, track_idx))
                if det_idx in unmatched_dets:
                    unmatched_dets.remove(det_idx)
                if track_idx in unmatched_tracks:
                    unmatched_tracks.remove(track_idx)

        return matched_indices, unmatched_dets, unmatched_tracks

    def _compute_cost_matrix(
        self,
        det_boxes: np.ndarray,
        track_boxes: np.ndarray,
        det_features: np.ndarray,
        track_features: np.ndarray
    ) -> np.ndarray:
        N, M = len(det_boxes), len(track_boxes)
        cost_matrix = np.zeros((N, M), dtype=np.float32)

        for i in range(N):
            for j in range(M):
                iou = self._compute_iou(det_boxes[i], track_boxes[j])
                iou_cost = 1.0 - iou

                det_center = np.array([
                    (det_boxes[i][0] + det_boxes[i][2]) / 2,
                    (det_boxes[i][1] + det_boxes[i][3]) / 2
                ])
                track_center = np.array([
                    (track_boxes[j][0] + track_boxes[j][2]) / 2,
                    (track_boxes[j][1] + track_boxes[j][3]) / 2
                ])
                
                spatial_distance = np.linalg.norm(det_center - track_center)
                
                bbox_size = max(
                    det_boxes[i][2] - det_boxes[i][0],
                    det_boxes[i][3] - det_boxes[i][1]
                )
                normalized_distance = spatial_distance / (bbox_size + 1e-6)
                
                if normalized_distance > 5.0:
                    cost_matrix[i, j] = 999.0
                    continue

                cosine_sim = self._cosine_similarity(det_features[i], track_features[j])
                reid_cost = 1.0 - cosine_sim

                if normalized_distance < 2.0:
                    cost_matrix[i, j] = 0.3 * iou_cost + 0.7 * reid_cost
                else:
                    cost_matrix[i, j] = 0.6 * iou_cost + 0.4 * reid_cost

        return cost_matrix

    def _compute_iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2

        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)

        if x2_i < x1_i or y2_i < y1_i:
            return 0.0

        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0.0

    def _cosine_similarity(self, feat1: np.ndarray, feat2: np.ndarray) -> float:
        dot = np.dot(feat1, feat2)
        norm1 = np.linalg.norm(feat1)
        norm2 = np.linalg.norm(feat2)
        return dot / (norm1 * norm2) if (norm1 > 0 and norm2 > 0) else 0.0

    def _greedy_matching(self, cost_matrix: np.ndarray) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        N, M = cost_matrix.shape
        matched_indices = []
        unmatched_dets = list(range(N))
        unmatched_tracks = list(range(M))
        threshold = 0.4

        while True:
            min_cost = float('inf')
            min_i, min_j = -1, -1

            for i in unmatched_dets:
                for j in unmatched_tracks:
                    if cost_matrix[i, j] < min_cost:
                        min_cost = cost_matrix[i, j]
                        min_i, min_j = i, j

            if min_cost < threshold:
                matched_indices.append((min_i, min_j))
                unmatched_dets.remove(min_i)
                unmatched_tracks.remove(min_j)
            else:
                break

        return matched_indices, unmatched_dets, unmatched_tracks

    def _get_next_track_id(self) -> int:
        if len(self.tracks) == 0:
            return 1
        return max(self.tracks.keys()) + 1

    def _cleanup_old_tracks(self, max_age: int = 90):
        tracks_to_delete = []
        for tid, track in self.tracks.items():
            if track.time_since_update > max_age:
                tracks_to_delete.append(tid)

        for tid in tracks_to_delete:
            del self.tracks[tid]


class Track:

    def __init__(
        self,
        track_id: int,
        box: np.ndarray,
        feature: np.ndarray,
        confidence: float,
        n_init: int = 8
    ):
        self.track_id = track_id
        self.box = box.copy()
        self.feature = feature.copy()
        self.confidence = confidence
        
        self.hits = 1
        self.time_since_update = 0
        self.n_init = n_init
        self.state = 'tentative'
        
        self.velocity = np.zeros(4, dtype=np.float32)

    def update(self, box: np.ndarray, feature: np.ndarray, confidence: float):
        new_velocity = box - self.box
        
        if self.hits > 1:
            velocity_change = np.linalg.norm(new_velocity - self.velocity)
            max_velocity_change = 50.0
            
            if velocity_change > max_velocity_change:
                alpha = 0.3
            else:
                alpha = 0.7
        else:
            alpha = 0.7
        
        self.velocity = 0.7 * new_velocity + 0.3 * self.velocity
        self.box = alpha * box + (1 - alpha) * self.box
        
        feature_alpha = 0.5
        self.feature = feature_alpha * feature + (1 - feature_alpha) * self.feature
        
        self.confidence = confidence
        self.hits += 1
        self.time_since_update = 0

        if self.hits >= self.n_init:
            self.state = 'confirmed'

    def mark_missed(self):
        self.time_since_update += 1
        
        if self.time_since_update <= 15:
            self.box = self.box + self.velocity * 0.3

    def is_confirmed(self) -> bool:
        return self.state == 'confirmed'

    def to_ltwh(self) -> List[float]:
        x1, y1, x2, y2 = self.box
        return [x1, y1, x2 - x1, y2 - y1]


class PlayerTracker:

    def __init__(self, config_path: str = None, alpha=0.15, persistence_frames=15, max_jump_distance=20.0):
        self.alpha = alpha
        self.persistence_frames = persistence_frames
        self.max_jump_distance = max_jump_distance
        
        if config_path and os.path.exists(config_path):
            self._load_config(config_path)
        
        self.smoothed_pos = {}
        self.last_detected = {}
        self.rejected_updates = {}
        self.position_history = {}
    
    def _load_config(self, config_path: str):
        try:
            current_section = None
            with open(config_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('['):
                        current_section = line.strip('[]')
                    elif '=' in line and current_section == 'player_tracker':
                        key, value = line.split('=', 1)
                        key = key.strip()
                        value = value.strip()
                        
                        if key == 'alpha':
                            self.alpha = float(value)
                        elif key == 'persistence-frames':
                            self.persistence_frames = int(value)
                        elif key == 'max-jump-distance':
                            self.max_jump_distance = float(value)
        except Exception as e:
            print(f"PlayerTracker config load error: {e}, using defaults")

    def update(self, player_id, court_pos, frame_idx):
        self.last_detected[player_id] = frame_idx

        if player_id not in self.smoothed_pos:
            self.smoothed_pos[player_id] = court_pos.copy()
            self.position_history[player_id] = [court_pos.copy()]
            self.rejected_updates[player_id] = 0
        else:
            old_pos = self.smoothed_pos[player_id]
            distance = np.linalg.norm(court_pos - old_pos)

            if distance > self.max_jump_distance:
                self.rejected_updates[player_id] = self.rejected_updates.get(player_id, 0) + 1

                if self.rejected_updates[player_id] >= 3:
                    self.smoothed_pos[player_id] = court_pos.copy()
                    self.position_history[player_id] = [court_pos.copy()]
                    self.rejected_updates[player_id] = 0

                return

            self.rejected_updates[player_id] = 0

            if distance < 3.0:
                adaptive_alpha = self.alpha * 0.5
            elif distance < 8.0:
                adaptive_alpha = self.alpha
            else:
                adaptive_alpha = min(self.alpha * 1.5, 0.20)

            self.smoothed_pos[player_id] = (
                adaptive_alpha * court_pos + (1 - adaptive_alpha) * old_pos
            )

            if player_id not in self.position_history:
                self.position_history[player_id] = []
            self.position_history[player_id].append(court_pos.copy())
            if len(self.position_history[player_id]) > 5:
                self.position_history[player_id].pop(0)

    def get_smoothed_position(self, player_id, frame_idx):
        if player_id not in self.smoothed_pos:
            return None

        frames_since_detection = frame_idx - self.last_detected.get(
            player_id, frame_idx
        )

        if frames_since_detection > self.persistence_frames:
            return None

        return self.smoothed_pos[player_id]

    def get_all_active_players(self, frame_idx):
        active = {}
        for player_id in list(self.last_detected.keys()):
            pos = self.get_smoothed_position(player_id, frame_idx)
            if pos is not None:
                active[player_id] = pos
            else:
                if (
                    frame_idx - self.last_detected.get(player_id, frame_idx)
                    > self.persistence_frames * 2
                ):
                    if player_id in self.smoothed_pos:
                        del self.smoothed_pos[player_id]
                    if player_id in self.last_detected:
                        del self.last_detected[player_id]
        return active


def create_tracker(config_path: str) -> Tracker:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Tracker config file not found: {config_path}")

    return Tracker(config_path)

