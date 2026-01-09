import cv2
import numpy as np

COURT_LENGTH = 94.0
COURT_WIDTH = 50.0
COURT_KEYPOINTS = np.array(
    [
        [0, 0],  # 0: left top corner
        [0, 3],  # 1: left baseline top
        [0, 17],  # 2: left paint top-left
        [0, 33],  # 3: left paint bottom-left
        [0, 47],  # 4: left baseline bottom
        [0, 50],  # 5: left bottom corner
        [5.25, 25],  # 6: left hoop
        [19, 17],  # 7: left FT top corner
        [19, 33],  # 8: left FT bottom corner
        [19, 19],  # 9: left FT circle top
        [19, 25],  # 10: left FT circle center
        [19, 31],  # 11: left FT circle bottom
        [14, 3],  # 12: left 3-point arc top
        [22, 17],  # 13: left elbow
        [14, 47],  # 14: left 3-point arc bottom
        [47, 0],  # 15: midcourt top
        [47, 25],  # 16: midcourt center
        [47, 50],  # 17: midcourt bottom
        [61, 3],  # 18: right top edge
        [72, 17],  # 19: right elbow
        [61, 47],  # 20: right bottom edge
        [75, 17],  # 21: right paint top-left
        [75, 25],  # 22: right FT center
        [75, 33],  # 23: right paint bottom-left
        [75, 17],  # 24: right FT top corner
        [80, 47],  # 25: right 3-point arc bottom
        [88.75, 25],  # 26: right hoop
        [94, 0],  # 27: right top corner
        [94, 3],  # 28: right baseline top
        [94, 17],  # 29: right paint top-right
        [94, 33],  # 30: right paint bottom-right
        [94, 47],  # 31: right baseline bottom
        [94, 50],  # 32: right bottom corner
    ],
    dtype=np.float32,
)

NEIGHBORS = {
    0: [1, 2, 15],
    1: [0, 2, 7, 12],
    2: [1, 3, 7],
    3: [2, 4, 8],
    4: [3, 5, 14],
    5: [4, 3, 17],
    6: [2, 3, 7, 8, 10],
    7: [1, 2, 9, 13],
    8: [3, 4, 11, 14],
    9: [7, 10],
    10: [6, 9, 11, 13],
    11: [8, 10],
    12: [1, 7, 13, 15],
    13: [7, 10, 12, 14],
    14: [4, 8, 13, 17],
    15: [0, 12, 16, 18, 27],
    16: [15, 17, 10, 22],
    17: [5, 14, 16, 20, 32],
    18: [15, 19, 24, 27],
    19: [18, 20, 22, 24],
    20: [17, 19, 23, 25],
    21: [22, 24, 29],
    22: [16, 19, 21, 23, 26],
    23: [20, 22, 30],
    24: [18, 19, 21, 28],
    25: [20, 23, 31, 32],
    26: [21, 22, 23, 29, 30],
    27: [15, 18, 28],
    28: [24, 27, 29],
    29: [21, 26, 28, 30],
    30: [23, 26, 29, 31],
    31: [25, 30, 32],
    32: [17, 25, 31],
}


def find_homography(image_pts, conf, min_points=4):
    valid = conf > 0.5
    if valid.sum() < min_points:
        return None

    src = COURT_KEYPOINTS[valid]
    dst = image_pts[valid]

    H, _ = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    return H


def project_points(H, court_pts):
    pts = np.hstack([court_pts, np.ones((len(court_pts), 1))])
    proj = (H @ pts.T).T
    return proj[:, :2] / proj[:, 2:3]


def image_to_court(H, image_pts):
    H_inv = np.linalg.inv(H)
    if image_pts.ndim == 1:
        image_pts = image_pts.reshape(1, -1)
    pts = np.hstack([image_pts, np.ones((len(image_pts), 1))])
    proj = (H_inv @ pts.T).T
    return proj[:, :2] / proj[:, 2:3]


def process_keypoints(last_kpts, last_conf, H, w, h):
    detected_indices = set(i for i in range(33) if last_conf[i] > 0.5)

    detected_pts = np.array([last_kpts[i] for i in detected_indices]) if detected_indices else None

    all_pts = []
    for i in range(33):
        if i in detected_indices:
            all_pts.append(last_kpts[i].copy())
        elif H is not None:
            has_detected_neighbor = any(
                n in detected_indices for n in NEIGHBORS.get(i, [])
            )
            if has_detected_neighbor:
                proj = project_points(H, COURT_KEYPOINTS[i : i + 1])[0]

                if not (-w < proj[0] < 2 * w and -h < proj[1] < 2 * h):
                    all_pts.append(None)
                    continue

                if detected_pts is not None and len(detected_pts) > 0:
                    distances = np.linalg.norm(detected_pts - proj, axis=1)
                    min_dist = np.min(distances)

                    max_allowed_dist = np.sqrt(w**2 + h**2) * 0.6
                    if min_dist > max_allowed_dist:
                        all_pts.append(None)
                        continue

                all_pts.append(proj)
            else:
                all_pts.append(None)
        else:
            all_pts.append(None)

    return all_pts, detected_indices


def smooth_keypoints(all_pts, smoothed_pts, alpha=0.4, w=None, h=None):
    if smoothed_pts is None:
        return [pt.copy() if pt is not None else None for pt in all_pts]

    for i in range(33):
        if all_pts[i] is not None:
            if smoothed_pts[i] is not None:
                dist = np.linalg.norm(all_pts[i] - smoothed_pts[i])
                if dist < 100:
                    smoothed_pts[i] = alpha * all_pts[i] + (1 - alpha) * smoothed_pts[i]
                else:
                    smoothed_pts[i] = all_pts[i].copy()
            else:
                smoothed_pts[i] = all_pts[i].copy()
        else:
            smoothed_pts[i] = None

    if w is not None and h is not None:
        for i in range(33):
            if smoothed_pts[i] is not None:
                x, y = smoothed_pts[i]
                if not (0 <= x < w and 0 <= y < h):
                    smoothed_pts[i] = None

    return smoothed_pts


def create_roi(all_pts, w, h):
    valid_pts = []
    for pt in all_pts:
        if pt is None:
            continue
        x, y = pt
        if -w < x < 2 * w and -h < y < 2 * h:
            x = np.clip(x, 0, w - 1)
            y = np.clip(y, 0, h - 1)
            valid_pts.append([x, y])

    if len(valid_pts) < 4:
        return None

    pts_array = np.array(valid_pts, dtype=np.float32)
    hull = cv2.convexHull(pts_array).astype(np.int32)
    return hull


def smooth_roi(current_roi, last_roi, alpha=0.3, max_frames_without_roi=10):
    if last_roi is None:
        if current_roi is not None:
            return current_roi, (current_roi, 0)
        return None, None

    last_hull, frames_without_roi = last_roi

    if current_roi is not None:
        if len(current_roi) != len(last_hull):
            return current_roi, (current_roi, 0)

        smoothed_hull = (alpha * current_roi + (1 - alpha) * last_hull).astype(np.int32)
        return smoothed_hull, (smoothed_hull, 0)

    else:
        frames_without_roi += 1

        if frames_without_roi < max_frames_without_roi:
            return last_hull, (last_hull, frames_without_roi)
        else:
            return None, None


def draw_roi(frame, hull):
    overlay = frame.copy()
    cv2.fillPoly(overlay, [hull], (0, 180, 0))
    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
    cv2.polylines(frame, [hull], True, (0, 255, 0), 2)


def draw_keypoints(frame, all_pts, last_conf, w, h):
    for i in range(33):
        if all_pts[i] is None:
            continue
        x, y = all_pts[i]
        if 0 <= x < w and 0 <= y < h:
            color = (0, 255, 0) if last_conf[i] > 0.5 else (0, 255, 255)
            cv2.circle(frame, (int(x), int(y)), 5, color, -1)
