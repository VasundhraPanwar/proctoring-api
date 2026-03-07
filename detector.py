"""
detector.py — Core cheating detection engine
Analyzes a video file frame by frame using MediaPipe + YOLO
Returns a structured JSON result with violations + cheating score
"""

import cv2
import time
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
import mediapipe as mp

# ── MediaPipe setup ────────────────────────────────────────────────────────────
mp_face_mesh    = mp.solutions.face_mesh
mp_face_detect  = mp.solutions.face_detection

# MediaPipe landmark indices
LEFT_EYE_INDICES  = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
RIGHT_EYE_INDICES = [33,  7,   163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
MOUTH_INDICES     = [61, 291, 39, 181, 0, 17, 269, 405]

# EAR (Eye Aspect Ratio) landmark pairs for blink/closure
LEFT_EAR_PAIRS  = [(385,380),(387,373),(386,374)]
RIGHT_EAR_PAIRS = [(159,145),(157,153),(158,154)]
LEFT_EAR_H      = (362, 263)
RIGHT_EAR_H     = (33,  133)

# Iris landmarks for gaze
LEFT_IRIS  = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]


@dataclass
class Violation:
    frame:   int
    time_s:  float
    type:    str
    detail:  str


@dataclass
class DetectionResult:
    student_id:       str
    video_path:       str
    duration_s:       float
    total_frames:     int
    processed_frames: int
    violations:       list = field(default_factory=list)
    cheating_score:   int  = 0
    risk_level:       str  = "LOW"
    counts:           dict = field(default_factory=dict)
    timeline:         list = field(default_factory=list)

    def to_dict(self):
        return {
            "student_id":       self.student_id,
            "video_path":       self.video_path,
            "duration_s":       round(self.duration_s, 2),
            "total_frames":     self.total_frames,
            "processed_frames": self.processed_frames,
            "cheating_score":   self.cheating_score,
            "risk_level":       self.risk_level,
            "total_violations": len(self.violations),
            "counts":           self.counts,
            "violations":       [v.__dict__ for v in self.violations[:100]],
            "timeline":         self.timeline,
        }


def _ear(landmarks, pairs, h_pair, w, h):
    """Eye Aspect Ratio — lower = more closed."""
    def pt(idx):
        lm = landmarks[idx]
        return np.array([lm.x * w, lm.y * h])
    num = sum(np.linalg.norm(pt(a) - pt(b)) for a, b in pairs)
    den = np.linalg.norm(pt(h_pair[0]) - pt(h_pair[1])) * len(pairs)
    return num / (den + 1e-6)


def _iris_gaze(landmarks, iris_idx, eye_h_pair, w, h):
    """
    Returns (gaze_x, gaze_y) as ratio within the eye bounding box.
    0.5,0.5 = center. Deviations indicate looking away.
    """
    def pt(idx):
        lm = landmarks[idx]
        return np.array([lm.x * w, lm.y * h])

    iris_center = np.mean([pt(i) for i in iris_idx], axis=0)
    eye_left    = pt(eye_h_pair[0])
    eye_right   = pt(eye_h_pair[1])
    eye_width   = np.linalg.norm(eye_right - eye_left) + 1e-6
    gaze_x      = (iris_center[0] - eye_left[0]) / eye_width
    return gaze_x  # 0=far left, 1=far right, ~0.5=center


def _mouth_open_ratio(landmarks, w, h):
    """Mouth aspect ratio — higher = more open."""
    def pt(idx):
        lm = landmarks[idx]
        return np.array([lm.x * w, lm.y * h])
    top    = pt(13)
    bottom = pt(14)
    left   = pt(61)
    right  = pt(291)
    return np.linalg.norm(top - bottom) / (np.linalg.norm(left - right) + 1e-6)


class CheatingDetector:
    def __init__(
        self,
        video_path: str,
        student_id: str = "",
        # Detection thresholds (tunable)
        process_every_n_frames: int  = 3,
        ear_threshold:          float = 0.20,   # below = eye closed / absent
        gaze_threshold:         float = 0.30,   # deviation from 0.5 center
        mouth_threshold:        float = 0.12,   # above = talking
        mouth_consec_frames:    int   = 4,
        gaze_consec_frames:     int   = 5,
        face_absent_consec:     int   = 8,
        object_conf_threshold:  float = 0.45,
        cooldown_s:             float = 3.0,    # min seconds between same violation type
    ):
        self.video_path     = video_path
        self.student_id     = student_id or Path(video_path).stem
        self.process_every  = process_every_n_frames
        self.ear_thresh     = ear_threshold
        self.gaze_thresh    = gaze_threshold
        self.mouth_thresh   = mouth_threshold
        self.mouth_consec   = mouth_consec_frames
        self.gaze_consec    = gaze_consec_frames
        self.face_absent_c  = face_absent_consec
        self.obj_conf       = object_conf_threshold
        self.cooldown       = cooldown_s

        # YOLO for object detection (lazy-loaded)
        self._yolo = None

        # Violation cooldown tracker: type → last_time
        self._last_violation: dict[str, float] = {}

    def _load_yolo(self):
        if self._yolo is None:
            try:
                from ultralytics import YOLO
                self._yolo = YOLO("yolov8n.pt")  # auto-downloads on first run
            except Exception as e:
                print(f"[WARN] YOLO not available: {e}. Object detection disabled.")
                self._yolo = False
        return self._yolo

    def _can_add(self, vtype: str, current_time: float) -> bool:
        last = self._last_violation.get(vtype, -999)
        if current_time - last >= self.cooldown:
            self._last_violation[vtype] = current_time
            return True
        return False

    def analyze(self) -> DetectionResult:
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {self.video_path}")

        fps         = cap.get(cv2.CAP_PROP_FPS) or 25
        total_f     = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_s  = total_f / fps
        w           = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h           = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        result = DetectionResult(
            student_id   = self.student_id,
            video_path   = self.video_path,
            duration_s   = duration_s,
            total_frames = total_f,
            processed_frames = 0,
        )

        violations: list[Violation] = []

        # ── Rolling counters ───────────────────────────────────────────────────
        mouth_open_count  = 0
        gaze_off_count    = 0
        face_absent_count = 0

        # ── Timeline: 10-second buckets ────────────────────────────────────────
        bucket_size  = 10  # seconds
        n_buckets    = max(1, int(duration_s / bucket_size) + 1)
        timeline     = [{"time_s": i * bucket_size, "violations": 0} for i in range(n_buckets)]

        yolo = self._load_yolo()

        with mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=4,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as face_mesh, \
        mp_face_detect.FaceDetection(
            model_selection=0,
            min_detection_confidence=0.5,
        ) as face_det:

            frame_idx = 0
            processed = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_idx += 1
                current_time = frame_idx / fps

                if frame_idx % self.process_every != 0:
                    continue

                processed += 1
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                bucket = min(int(current_time / bucket_size), n_buckets - 1)

                def add_v(vtype, detail):
                    if self._can_add(vtype, current_time):
                        v = Violation(frame_idx, round(current_time, 2), vtype, detail)
                        violations.append(v)
                        timeline[bucket]["violations"] += 1

                # ── 1. Face detection (count faces) ────────────────────────────
                fd_result = face_det.process(rgb)
                n_faces   = len(fd_result.detections) if fd_result.detections else 0

                if n_faces == 0:
                    face_absent_count += 1
                    if face_absent_count >= self.face_absent_c:
                        add_v("face_not_visible", "Face absent from frame")
                        face_absent_count = 0
                else:
                    face_absent_count = 0

                if n_faces > 1:
                    add_v("multiple_faces", f"{n_faces} faces detected in frame")

                # ── 2. Face mesh — eyes + mouth ────────────────────────────────
                if n_faces >= 1:
                    mesh_result = face_mesh.process(rgb)

                    if mesh_result.multi_face_landmarks:
                        lm = mesh_result.multi_face_landmarks[0].landmark

                        # Eye Aspect Ratio — closed / absent
                        left_ear  = _ear(lm, LEFT_EAR_PAIRS,  LEFT_EAR_H,  w, h)
                        right_ear = _ear(lm, RIGHT_EAR_PAIRS, RIGHT_EAR_H, w, h)
                        avg_ear   = (left_ear + right_ear) / 2

                        # Gaze via iris position
                        left_gaze  = _iris_gaze(lm, LEFT_IRIS,  LEFT_EAR_H,  w, h)
                        right_gaze = _iris_gaze(lm, RIGHT_IRIS, RIGHT_EAR_H, w, h)
                        avg_gaze   = (left_gaze + right_gaze) / 2
                        gaze_dev   = abs(avg_gaze - 0.5)

                        if gaze_dev > self.gaze_thresh and avg_ear > self.ear_thresh:
                            gaze_off_count += 1
                            if gaze_off_count >= self.gaze_consec:
                                direction = "left" if avg_gaze < 0.5 else "right"
                                add_v("eye_movement", f"Eyes looking {direction} (deviation={gaze_dev:.2f})")
                                gaze_off_count = 0
                        else:
                            gaze_off_count = 0

                        # Mouth open ratio
                        mar = _mouth_open_ratio(lm, w, h)
                        if mar > self.mouth_thresh:
                            mouth_open_count += 1
                            if mouth_open_count >= self.mouth_consec:
                                add_v("mouth_movement", f"Mouth open/talking (MAR={mar:.2f})")
                                mouth_open_count = 0
                        else:
                            mouth_open_count = 0

                # ── 3. Object detection (every 15 frames to save time) ─────────
                if yolo and frame_idx % 15 == 0:
                    try:
                        yolo_results = yolo(rgb, verbose=False, conf=self.obj_conf)
                        prohibited   = {"cell phone", "book", "laptop", "tablet", "earphone", "airpods"}
                        for r in yolo_results:
                            for box in r.boxes:
                                cls_name = yolo.model.names[int(box.cls)].lower()
                                if any(p in cls_name for p in prohibited):
                                    add_v("prohibited_object", f"Detected: {cls_name}")
                    except Exception:
                        pass

        cap.release()

        # ── Compute final score ────────────────────────────────────────────────
        counts = {
            "face_not_visible":  sum(1 for v in violations if v.type == "face_not_visible"),
            "multiple_faces":    sum(1 for v in violations if v.type == "multiple_faces"),
            "eye_movement":      sum(1 for v in violations if v.type == "eye_movement"),
            "mouth_movement":    sum(1 for v in violations if v.type == "mouth_movement"),
            "prohibited_object": sum(1 for v in violations if v.type == "prohibited_object"),
        }

        raw = (
            counts["multiple_faces"]    * 18 +
            counts["prohibited_object"] * 15 +
            counts["face_not_visible"]  * 12 +
            counts["eye_movement"]      *  6 +
            counts["mouth_movement"]    *  5
        )
        score = min(100, raw)
        risk  = "HIGH" if score >= 60 else "MEDIUM" if score >= 25 else "LOW"

        result.violations       = violations
        result.processed_frames = processed
        result.cheating_score   = score
        result.risk_level       = risk
        result.counts           = counts
        result.timeline         = timeline

        return result
