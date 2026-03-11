"""
detector.py — Interview Cheating Detection Engine v4

Pipeline:
  1. Extract audio → Whisper transcription (full transcript stored)
  2. Claude identifies interviewee NAME from transcript
  3. OCR scans early frames → finds name label on screen (Google Meet/Zoom UI)
  4. Face ABOVE that name label = interviewee, get their bounding box
  5. Track only that face — detect repeated same-direction gaze pattern
  6. Return structured JSON with full transcript + gaze analysis
"""

import os
import cv2
import json
import tempfile
import subprocess
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from collections import Counter, deque
import mediapipe as mp
import anthropic
import openai

# ── API clients ────────────────────────────────────────────────────────────────
anthropic_client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
openai_client    = openai.OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

# ── MediaPipe ─────────────────────────────────────────────────────────────────
mp_face_mesh   = mp.solutions.face_mesh
mp_face_detect = mp.solutions.face_detection

LEFT_IRIS   = [474, 475, 476, 477]
RIGHT_IRIS  = [469, 470, 471, 472]
LEFT_EAR_H  = (362, 263)
RIGHT_EAR_H = (33,  133)

MODEL_POINTS = np.array([
    (0.0,    0.0,    0.0),
    (0.0,   -330.0, -65.0),
    (-225.0, 170.0, -135.0),
    (225.0,  170.0, -135.0),
    (-150.0, -150.0, -125.0),
    (150.0,  -150.0, -125.0),
], dtype=np.float64)


@dataclass
class Violation:
    frame:    int
    time_s:   float
    type:     str
    detail:   str
    severity: str


@dataclass
class DetectionResult:
    student_id:          str
    video_path:          str
    duration_s:          float
    total_frames:        int
    processed_frames:    int
    interviewee_name:    str  = "unknown"
    interviewee_region:  dict = field(default_factory=dict)
    transcript_full:     str  = ""
    transcript_summary:  str  = ""
    interview_topic:     str  = ""
    violations:          list = field(default_factory=list)
    gaze_pattern:        dict = field(default_factory=dict)
    cheating_score:      int  = 0
    risk_level:          str  = "LOW"
    counts:              dict = field(default_factory=dict)
    timeline:            list = field(default_factory=list)
    summary:             str  = ""

    def to_dict(self):
        return {
            "student_id":         self.student_id,
            "video_path":         self.video_path,
            "duration_s":         round(self.duration_s, 2),
            "total_frames":       self.total_frames,
            "processed_frames":   self.processed_frames,
            "interviewee_name":   self.interviewee_name,
            "interviewee_region": self.interviewee_region,
            "transcript_full":    self.transcript_full,
            "transcript_summary": self.transcript_summary,
            "interview_topic":    self.interview_topic,
            "cheating_score":     self.cheating_score,
            "risk_level":         self.risk_level,
            "total_violations":   len(self.violations),
            "gaze_pattern":       self.gaze_pattern,
            "counts":             self.counts,
            "violations":         [v.__dict__ for v in self.violations[:100]],
            "timeline":           self.timeline,
            "summary":            self.summary,
        }


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1 — Extract audio
# ══════════════════════════════════════════════════════════════════════════════
def extract_audio(video_path: str) -> Optional[str]:
    try:
        audio_path = tempfile.mktemp(suffix=".mp3")
        result = subprocess.run([
            "ffmpeg", "-i", video_path,
            "-vn", "-acodec", "libmp3lame",
            "-ar", "16000", "-ac", "1", "-q:a", "4",
            audio_path, "-y", "-loglevel", "error"
        ], capture_output=True, timeout=300)
        if result.returncode == 0 and Path(audio_path).exists():
            print(f"[AUDIO] Extracted: {Path(audio_path).stat().st_size/1024/1024:.1f}MB")
            return audio_path
        print(f"[AUDIO] ffmpeg failed: {result.stderr.decode()}")
        return None
    except Exception as e:
        print(f"[AUDIO] Exception: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
# STEP 2 — Transcribe with Whisper (chunked for long videos)
# ══════════════════════════════════════════════════════════════════════════════
def transcribe_audio(audio_path: str) -> Optional[dict]:
    try:
        size_mb = Path(audio_path).stat().st_size / 1024 / 1024
        print(f"[WHISPER] Transcribing {size_mb:.1f}MB audio...")

        # If > 20MB, compress further
        if size_mb > 20:
            compressed = tempfile.mktemp(suffix=".mp3")
            subprocess.run([
                "ffmpeg", "-i", audio_path,
                "-ar", "8000", "-ac", "1", "-q:a", "9",
                compressed, "-y", "-loglevel", "error"
            ], capture_output=True, timeout=120)
            if Path(compressed).exists():
                audio_path = compressed
                print(f"[WHISPER] Compressed to {Path(compressed).stat().st_size/1024/1024:.1f}MB")

        with open(audio_path, "rb") as f:
            response = openai_client.audio.transcriptions.create(
                model="whisper-1",
                file=f,
                response_format="verbose_json",
                timestamp_granularities=["segment"]
            )

        segments = []
        for seg in (response.segments or []):
            segments.append({
                "start": round(seg.start, 2),
                "end":   round(seg.end, 2),
                "text":  seg.text.strip(),
            })

        print(f"[WHISPER] {len(segments)} segments, {len(response.text)} chars, lang={response.language}")
        return {
            "full_text": response.text,
            "segments":  segments,
            "language":  response.language,
        }
    except Exception as e:
        print(f"[WHISPER] Failed: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
# STEP 3 — Claude identifies interviewee name + summary
# ══════════════════════════════════════════════════════════════════════════════
def identify_interviewee_name(transcript: dict) -> dict:
    try:
        full_text = transcript.get("full_text", "")
        if not full_text or len(full_text) < 30:
            return {"interviewee_name": None, "summary": "Too short", "interview_topic": "unknown"}

        prompt = f"""You are analyzing a job interview transcript.

TRANSCRIPT:
{full_text[:5000]}

1. Who is the INTERVIEWEE? (they answer questions about their background/skills/experience)
2. Who is the INTERVIEWER? (they ask questions)
3. If the interviewee's name is mentioned anywhere in the conversation, extract it.
   Names are often mentioned at the start: "Hi I'm John" or "Can you introduce yourself, Priya?"
4. What is the interview about? (job role / skill being evaluated)
5. Write a 3-5 sentence summary of what was discussed.

Respond ONLY in this exact JSON format, no other text:
{{
  "interviewee_name": "first name or full name if found, else null",
  "interviewer_name": "first name or full name if found, else null",
  "interview_topic": "e.g. Oracle EBS Functional Consultant",
  "summary": "3-5 sentence summary of the interview",
  "confidence": "HIGH or MEDIUM or LOW",
  "name_mention": "quote from transcript where name was mentioned, or null"
}}"""

        response = anthropic_client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}]
        )

        text = response.content[0].text.strip().replace("```json","").replace("```","").strip()
        parsed = json.loads(text)
        print(f"[CLAUDE] Interviewee: {parsed.get('interviewee_name')} | Topic: {parsed.get('interview_topic')} | Confidence: {parsed.get('confidence')}")
        return parsed

    except Exception as e:
        print(f"[CLAUDE] Failed: {e}")
        return {"interviewee_name": None, "summary": f"Error: {e}", "interview_topic": "unknown"}


# ══════════════════════════════════════════════════════════════════════════════
# STEP 4 — OCR: find name label on screen, get face region above it
# ══════════════════════════════════════════════════════════════════════════════
def find_interviewee_region_by_ocr(video_path: str, interviewee_name: Optional[str], frame_w: int, frame_h: int) -> dict:
    """
    Scan early frames with OCR (pytesseract) to find the name label.
    Google Meet/Zoom renders name as white text near the bottom of each video tile.
    Returns bounding box {x1, y1, x2, y2} of the interviewee's tile.
    Falls back to largest-face detection if OCR fails or name not found.
    """

    # Try OCR first if we have a name
    if interviewee_name:
        try:
            import pytesseract
            from PIL import Image

            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 25
            found_region = None

            # Scan frames at 0s, 30s, 60s, 120s, 180s
            for t in [0, 30, 60, 120, 180]:
                cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
                ret, frame = cap.read()
                if not ret:
                    continue

                # OCR on bottom 30% of frame (where name labels appear)
                bottom_strip = frame[int(frame_h * 0.7):, :]
                gray = cv2.cvtColor(bottom_strip, cv2.COLOR_BGR2GRAY)
                # Threshold to make white text pop
                _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)

                # Get OCR data with bounding boxes
                ocr_data = pytesseract.image_to_data(
                    Image.fromarray(thresh),
                    output_type=pytesseract.Output.DICT,
                    config='--psm 11'
                )

                # Search for interviewee name (first name match is enough)
                first_name = interviewee_name.split()[0].lower()
                for i, word in enumerate(ocr_data['text']):
                    if first_name in word.lower() and len(word) >= 3:
                        # Found the name label
                        nx = ocr_data['left'][i]
                        ny = ocr_data['top'][i] + int(frame_h * 0.7)
                        nw = ocr_data['width'][i]
                        nh = ocr_data['height'][i]

                        print(f"[OCR] Found '{word}' at frame t={t}s, pos=({nx},{ny})")

                        # The video tile extends upward from the name label
                        # Estimate tile: name is at bottom, face fills the tile above
                        # Typical Meet tile ratio ~16:9, so tile height ≈ tile width * 9/16
                        # Expand region: tile goes from (nx - padding) to (nx + tile_width) wide
                        # and from (ny - tile_height) to (ny + label_height) tall

                        # Find tile boundaries by looking for the tile container
                        # Heuristic: expand outward from name until we hit dark border
                        tile_x1 = max(0, nx - 50)
                        tile_x2 = min(frame_w, nx + nw + 200)
                        tile_width = tile_x2 - tile_x1
                        tile_height = int(tile_width * 9 / 16)
                        tile_y1 = max(0, ny - tile_height)
                        tile_y2 = min(frame_h, ny + nh + 10)

                        found_region = {
                            "x1": tile_x1, "y1": tile_y1,
                            "x2": tile_x2, "y2": tile_y2,
                            "method": "ocr",
                            "name_found": word,
                        }
                        break

                if found_region:
                    break

            cap.release()

            if found_region:
                print(f"[OCR] Region: {found_region}")
                return found_region

        except ImportError:
            print("[OCR] pytesseract not installed, falling back to largest-face method")
        except Exception as e:
            print(f"[OCR] Failed: {e}, falling back to largest-face method")

    # ── Fallback: find largest face in early frames ────────────────────────────
    print("[REGION] Using largest-face fallback method")
    return find_largest_face_region(video_path, frame_w, frame_h)


def find_largest_face_region(video_path: str, frame_w: int, frame_h: int) -> dict:
    """
    Scan early frames, find the largest face (= main speaker = interviewee).
    Returns a padded bounding box around that face.
    """
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25

    best_area  = 0
    best_box   = None

    with mp_face_detect.FaceDetection(model_selection=1, min_detection_confidence=0.5) as fd:
        # Sample 10 frames spread across first 2 minutes
        sample_times = [5, 15, 30, 45, 60, 75, 90, 105, 120]
        for t in sample_times:
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
            ret, frame = cap.read()
            if not ret:
                continue

            rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = fd.process(rgb)

            if not results.detections:
                continue

            for det in results.detections:
                bb = det.location_data.relative_bounding_box
                w  = int(bb.width  * frame_w)
                h  = int(bb.height * frame_h)
                area = w * h
                if area > best_area:
                    best_area = area
                    x = int(bb.xmin * frame_w)
                    y = int(bb.ymin * frame_h)
                    best_box = (x, y, w, h)

    cap.release()

    if best_box:
        x, y, w, h = best_box
        # Pad generously to capture full head + shoulders
        pad_x = int(w * 0.6)
        pad_y = int(h * 0.8)
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(frame_w, x + w + pad_x)
        y2 = min(frame_h, y + h + pad_y)
        print(f"[REGION] Largest face at ({x},{y}) size={w}x{h} → region ({x1},{y1})-({x2},{y2})")
        return {"x1": x1, "y1": y1, "x2": x2, "y2": y2, "method": "largest_face"}

    # Last resort: full frame
    print("[REGION] No face found, using full frame")
    return {"x1": 0, "y1": 0, "x2": frame_w, "y2": frame_h, "method": "full_frame"}


# ══════════════════════════════════════════════════════════════════════════════
# Head pose + gaze helpers
# ══════════════════════════════════════════════════════════════════════════════
def get_head_pose(landmarks, w, h):
    pts = np.array([
        (landmarks[1].x*w,   landmarks[1].y*h),
        (landmarks[152].x*w, landmarks[152].y*h),
        (landmarks[33].x*w,  landmarks[33].y*h),
        (landmarks[263].x*w, landmarks[263].y*h),
        (landmarks[61].x*w,  landmarks[61].y*h),
        (landmarks[291].x*w, landmarks[291].y*h),
    ], dtype=np.float64)
    cam  = np.array([[w,0,w/2],[0,w,h/2],[0,0,1]], dtype=np.float64)
    dist = np.zeros((4,1))
    _, rvec, _ = cv2.solvePnP(MODEL_POINTS, pts, cam, dist, flags=cv2.SOLVEPNP_ITERATIVE)
    rmat, _    = cv2.Rodrigues(rvec)
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
    return angles  # pitch, yaw, roll


def get_iris_gaze(landmarks, iris_idx, eye_h_pair, w, h):
    def pt(i): return np.array([landmarks[i].x*w, landmarks[i].y*h])
    iris  = np.mean([pt(i) for i in iris_idx], axis=0)
    width = np.linalg.norm(pt(eye_h_pair[1]) - pt(eye_h_pair[0])) + 1e-6
    return (iris[0] - pt(eye_h_pair[0])[0]) / width


def classify_gaze_direction(yaw: float, pitch: float, gaze_x: float) -> str:
    """
    Returns: LEFT, RIGHT, DOWN, UP, CENTER
    Uses both head pose (yaw/pitch) and iris position (gaze_x)
    """
    # Head pose takes priority for big movements
    if abs(yaw) > 20:
        return "LEFT" if yaw < 0 else "RIGHT"
    if pitch < -12:
        return "DOWN"
    if pitch > 15:
        return "UP"

    # Iris gaze for subtle movements
    if gaze_x < 0.40:
        return "LEFT"
    if gaze_x > 0.60:
        return "RIGHT"

    return "CENTER"


# ══════════════════════════════════════════════════════════════════════════════
# STEP 5 — Detect repeated same-direction gaze patterns
# ══════════════════════════════════════════════════════════════════════════════
def detect_direction_patterns(gaze_log: list, min_occurrences: int = 3, window_s: float = 60.0) -> list:
    """
    Finds repeated same-direction looks within rolling time windows.
    
    e.g. if person looks LEFT 5 times within 60 seconds → suspicious pattern
    Random glances in different directions → not flagged
    
    Returns list of detected patterns with timestamps.
    """
    if not gaze_log:
        return []

    patterns = []

    # Use a rolling window approach
    # Group consecutive off-center frames into "glance events"
    glance_events = []
    current_dir   = None
    current_start = None
    CENTER_FRAMES_TO_RESET = 3  # need 3 center frames to end a glance
    center_streak = 0

    for entry in gaze_log:
        d = entry["direction"]
        t = entry["time_s"]

        if d == "CENTER" or d == "ABSENT":
            center_streak += 1
            if center_streak >= CENTER_FRAMES_TO_RESET and current_dir:
                # Glance ended
                glance_events.append({
                    "direction": current_dir,
                    "time_s":    current_start,
                })
                current_dir   = None
                current_start = None
        else:
            center_streak = 0
            if d != current_dir:
                if current_dir:
                    glance_events.append({
                        "direction": current_dir,
                        "time_s":    current_start,
                    })
                current_dir   = d
                current_start = t

    # Close last glance
    if current_dir:
        glance_events.append({"direction": current_dir, "time_s": current_start})

    print(f"[PATTERN] Total glance events: {len(glance_events)}")

    # Now find repeated same-direction glances within rolling windows
    for direction in ["LEFT", "RIGHT", "DOWN", "UP"]:
        dir_events = [e for e in glance_events if e["direction"] == direction]

        if len(dir_events) < min_occurrences:
            continue

        # Slide window across events
        flagged_windows = []
        for i in range(len(dir_events)):
            window_events = [
                e for e in dir_events
                if dir_events[i]["time_s"] <= e["time_s"] <= dir_events[i]["time_s"] + window_s
            ]
            if len(window_events) >= min_occurrences:
                t_start = window_events[0]["time_s"]
                t_end   = window_events[-1]["time_s"]

                # Don't double-flag overlapping windows
                already_flagged = any(
                    abs(f["time_s"] - t_start) < window_s / 2
                    for f in flagged_windows
                )
                if not already_flagged:
                    flagged_windows.append({
                        "direction": direction,
                        "time_s":    round(t_start, 2),
                        "count":     len(window_events),
                        "duration_s": round(t_end - t_start, 1),
                        "detail":    f"Looked {direction} {len(window_events)} times in {t_end-t_start:.0f}s window (t={t_start:.0f}s–{t_end:.0f}s) — possible {'notes on screen' if direction in ['LEFT','RIGHT'] else 'notes on desk/lap'}",
                    })

        if flagged_windows:
            print(f"[PATTERN] {direction}: {len(flagged_windows)} suspicious windows")
            patterns.extend(flagged_windows)

    return sorted(patterns, key=lambda p: p["time_s"])


# ══════════════════════════════════════════════════════════════════════════════
# Main detector class
# ══════════════════════════════════════════════════════════════════════════════
class InterviewCheatingDetector:
    def __init__(
        self,
        video_path:             str,
        student_id:             str   = "",
        process_every_n_frames: int   = 3,
        offscreen_duration_s:   float = 3.0,
        object_conf:            float = 0.40,
        cooldown_s:             float = 5.0,
        dir_min_occurrences:    int   = 3,
        dir_window_s:           float = 60.0,
    ):
        self.video_path    = video_path
        self.student_id    = student_id or Path(video_path).stem
        self.process_every = process_every_n_frames
        self.offscreen_dur = offscreen_duration_s
        self.obj_conf      = object_conf
        self.cooldown      = cooldown_s
        self.dir_min_occ   = dir_min_occurrences
        self.dir_window    = dir_window_s
        self._last_v: dict = {}
        self._yolo = None

    def _load_yolo(self):
        if self._yolo is None:
            try:
                from ultralytics import YOLO
                self._yolo = YOLO("yolov8n.pt")
            except Exception as e:
                print(f"[YOLO] Unavailable: {e}")
                self._yolo = False
        return self._yolo

    def _can_add(self, vtype, t):
        if t - self._last_v.get(vtype, -999) >= self.cooldown:
            self._last_v[vtype] = t
            return True
        return False

    def analyze(self) -> DetectionResult:
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open: {self.video_path}")
        fps        = cap.get(cv2.CAP_PROP_FPS) or 25
        total_f    = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_s = total_f / fps
        frame_w    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h    = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        print(f"\n{'='*60}")
        print(f"[ANALYZE] {self.student_id}")
        print(f"[ANALYZE] {frame_w}x{frame_h} | {duration_s:.0f}s | {total_f} frames @ {fps:.0f}fps")
        print(f"{'='*60}")

        result = DetectionResult(
            student_id=self.student_id, video_path=self.video_path,
            duration_s=duration_s, total_frames=total_f, processed_frames=0,
        )

        # ── PHASE 1: Audio → Whisper → Claude ─────────────────────────────────
        print(f"\n[PHASE 1] Audio transcription + interviewee identification")
        transcript_full    = ""
        transcript_summary = ""
        interview_topic    = ""
        interviewee_name   = None

        audio_path = extract_audio(self.video_path)
        if audio_path:
            transcript = transcribe_audio(audio_path)
            if transcript:
                transcript_full = transcript["full_text"]
                print(f"[PHASE 1] Asking Claude to identify interviewee...")
                info = identify_interviewee_name(transcript)
                interviewee_name   = info.get("interviewee_name")
                transcript_summary = info.get("summary", "")
                interview_topic    = info.get("interview_topic", "")
            try: Path(audio_path).unlink()
            except: pass

        result.interviewee_name   = interviewee_name or "unknown"
        result.transcript_full    = transcript_full
        result.transcript_summary = transcript_summary
        result.interview_topic    = interview_topic

        # ── PHASE 2: Find interviewee region ──────────────────────────────────
        print(f"\n[PHASE 2] Locating interviewee on screen")
        region = find_interviewee_region_by_ocr(
            self.video_path, interviewee_name, frame_w, frame_h
        )
        result.interviewee_region = region
        x1, y1, x2, y2 = region["x1"], region["y1"], region["x2"], region["y2"]
        crop_w = x2 - x1
        crop_h = y2 - y1
        print(f"[PHASE 2] Region: ({x1},{y1})→({x2},{y2}) size={crop_w}x{crop_h} method={region['method']}")

        # ── PHASE 3: Frame-by-frame detection ─────────────────────────────────
        print(f"\n[PHASE 3] CV detection on interviewee region")
        cap        = cv2.VideoCapture(self.video_path)
        violations = []
        gaze_log   = []

        bucket_size = 10
        n_buckets   = max(1, int(duration_s / bucket_size) + 1)
        timeline    = [{"time_s": i*bucket_size, "violations": 0} for i in range(n_buckets)]
        offscreen_start = None
        processed       = 0
        yolo            = self._load_yolo()

        with mp_face_mesh.FaceMesh(
            static_image_mode=False, max_num_faces=2,
            refine_landmarks=True,
            min_detection_confidence=0.5, min_tracking_confidence=0.5,
        ) as face_mesh, \
        mp_face_detect.FaceDetection(
            model_selection=0, min_detection_confidence=0.5,
        ) as face_det:

            frame_idx = 0
            while True:
                ret, frame = cap.read()
                if not ret: break
                frame_idx   += 1
                current_time = frame_idx / fps

                if frame_idx % self.process_every != 0: continue
                processed += 1

                # Crop to interviewee region
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0: continue
                rgb    = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                bucket = min(int(current_time / bucket_size), n_buckets - 1)

                def add_v(vtype, detail, severity="MEDIUM"):
                    if self._can_add(vtype, current_time):
                        violations.append(Violation(frame_idx, round(current_time,2), vtype, detail, severity))
                        timeline[bucket]["violations"] += 1

                # Face count
                fd      = face_det.process(rgb)
                n_faces = len(fd.detections) if fd.detections else 0

                if n_faces == 0:
                    gaze_log.append({"time_s": round(current_time,2), "direction": "ABSENT"})
                    add_v("interviewee_absent", "Interviewee not visible in region", "LOW")
                    continue

                if n_faces > 1:
                    add_v("third_person_detected",
                          f"{n_faces} faces detected — possible coaching", "HIGH")

                # Face mesh → gaze direction
                mesh = face_mesh.process(rgb)
                if mesh.multi_face_landmarks:
                    lm = mesh.multi_face_landmarks[0].landmark

                    try:
                        pitch, yaw, roll = get_head_pose(lm, crop_w, crop_h)
                    except:
                        pitch, yaw, roll = 0, 0, 0

                    try:
                        lg     = get_iris_gaze(lm, LEFT_IRIS,  LEFT_EAR_H,  crop_w, crop_h)
                        rg     = get_iris_gaze(lm, RIGHT_IRIS, RIGHT_EAR_H, crop_w, crop_h)
                        gaze_x = (lg + rg) / 2
                    except:
                        gaze_x = 0.5

                    direction = classify_gaze_direction(yaw, pitch, gaze_x)
                    gaze_log.append({"time_s": round(current_time,2), "direction": direction})

                    # Sustained off-screen gaze (immediate)
                    if direction != "CENTER":
                        if offscreen_start is None:
                            offscreen_start = current_time
                        elif current_time - offscreen_start >= self.offscreen_dur:
                            add_v("sustained_gaze",
                                  f"Looking {direction} continuously for {current_time-offscreen_start:.1f}s",
                                  "MEDIUM")
                            offscreen_start = None
                    else:
                        offscreen_start = None

                # Object detection every 15 frames
                if yolo and frame_idx % 15 == 0:
                    try:
                        for r in yolo(rgb, verbose=False, conf=self.obj_conf):
                            for box in r.boxes:
                                cls = yolo.model.names[int(box.cls)].lower()
                                if "phone" in cls:
                                    add_v("phone_detected", "Phone visible", "HIGH")
                                elif any(x in cls for x in ["book","laptop","tablet","earphone","airpod","headphone","notebook"]):
                                    add_v("prohibited_object", f"Detected: {cls}", "MEDIUM")
                    except: pass

        cap.release()
        print(f"[PHASE 3] Processed {processed} frames, {len(gaze_log)} gaze readings, {len(violations)} raw violations")

        # ── PHASE 4: Pattern analysis ──────────────────────────────────────────
        print(f"\n[PHASE 4] Gaze direction pattern analysis")
        patterns = detect_direction_patterns(
            gaze_log,
            min_occurrences = self.dir_min_occ,
            window_s        = self.dir_window,
        )

        direction_counts = Counter(g["direction"] for g in gaze_log)
        total_tracked    = len(gaze_log) or 1
        gaze_pattern = {
            "LEFT_pct":   round(direction_counts.get("LEFT",  0) / total_tracked * 100, 1),
            "RIGHT_pct":  round(direction_counts.get("RIGHT", 0) / total_tracked * 100, 1),
            "DOWN_pct":   round(direction_counts.get("DOWN",  0) / total_tracked * 100, 1),
            "CENTER_pct": round(direction_counts.get("CENTER",0) / total_tracked * 100, 1),
            "total_glances_logged": len(gaze_log),
            "repeated_patterns":    patterns,
        }

        print(f"[PHASE 4] LEFT={gaze_pattern['LEFT_pct']}% RIGHT={gaze_pattern['RIGHT_pct']}% DOWN={gaze_pattern['DOWN_pct']}% CENTER={gaze_pattern['CENTER_pct']}%")
        print(f"[PHASE 4] Suspicious patterns: {len(patterns)}")
        for p in patterns:
            print(f"          → {p['detail']}")

        # Add pattern violations
        for p in patterns:
            if self._can_add(f"pattern_{p['direction']}", p["time_s"]):
                violations.append(Violation(
                    0, p["time_s"],
                    "repeated_direction_pattern",
                    p["detail"],
                    "HIGH"
                ))

        # ── SCORING ───────────────────────────────────────────────────────────
        counts = {
            "third_person_detected":      sum(1 for v in violations if v.type=="third_person_detected"),
            "phone_detected":             sum(1 for v in violations if v.type=="phone_detected"),
            "prohibited_object":          sum(1 for v in violations if v.type=="prohibited_object"),
            "sustained_gaze":             sum(1 for v in violations if v.type=="sustained_gaze"),
            "repeated_direction_pattern": sum(1 for v in violations if v.type=="repeated_direction_pattern"),
            "interviewee_absent":         sum(1 for v in violations if v.type=="interviewee_absent"),
        }

        raw = (
            counts["third_person_detected"]      * 25 +
            counts["phone_detected"]             * 22 +
            counts["repeated_direction_pattern"] * 20 +
            counts["prohibited_object"]          * 15 +
            counts["sustained_gaze"]             *  8 +
            counts["interviewee_absent"]         *  3
        )

        # Boost if gaze strongly skewed to one direction (> 30% of time)
        dominant_pct = max(gaze_pattern["LEFT_pct"], gaze_pattern["RIGHT_pct"], gaze_pattern["DOWN_pct"])
        if dominant_pct > 30:
            boost = 15
            raw  += boost
            print(f"[SCORE] Direction dominance boost: +{boost} ({dominant_pct:.0f}% off-center)")

        score = min(100, raw)
        risk  = "HIGH" if score >= 60 else "MEDIUM" if score >= 25 else "LOW"

        parts = []
        if counts["third_person_detected"]>0:
            parts.append(f"3rd person {counts['third_person_detected']}x")
        if counts["phone_detected"]>0:
            parts.append(f"Phone {counts['phone_detected']}x")
        if counts["repeated_direction_pattern"]>0:
            dirs = [p["direction"] for p in patterns]
            dominant = Counter(dirs).most_common(1)[0][0] if dirs else "unknown"
            parts.append(f"Repeated {dominant} look {counts['repeated_direction_pattern']}x")
        if counts["prohibited_object"]>0:
            parts.append(f"Notes/objects {counts['prohibited_object']}x")
        if counts["sustained_gaze"]>0:
            parts.append(f"Sustained gaze off-screen {counts['sustained_gaze']}x")
        if gaze_pattern["LEFT_pct"] > 25:
            parts.append(f"Eyes LEFT {gaze_pattern['LEFT_pct']}% of interview")
        if gaze_pattern["RIGHT_pct"] > 25:
            parts.append(f"Eyes RIGHT {gaze_pattern['RIGHT_pct']}% of interview")
        if gaze_pattern["DOWN_pct"] > 20:
            parts.append(f"Eyes DOWN {gaze_pattern['DOWN_pct']}% of interview")

        summary = ", ".join(parts) if parts else "No suspicious activity detected"

        print(f"\n[RESULT] Score={score}/100 | Risk={risk}")
        print(f"[RESULT] {summary}")
        print(f"{'='*60}\n")

        result.violations       = violations
        result.processed_frames = processed
        result.cheating_score   = score
        result.risk_level       = risk
        result.counts           = counts
        result.gaze_pattern     = gaze_pattern
        result.timeline         = timeline
        result.summary          = summary
        return result
