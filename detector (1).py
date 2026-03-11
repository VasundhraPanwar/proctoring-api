"""
detector.py — Interview Cheating Detection Engine v3

Pipeline:
  1. Extract audio from video
  2. Whisper API → transcription + speaker diarization
  3. Claude API → identify which speaker is the interviewee
  4. Find interviewee face position (left/right half of screen)
  5. Track only that face for cheating violations
  6. Return structured JSON result
"""

import os
import cv2
import tempfile
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
import mediapipe as mp
import anthropic
import openai

# ── API clients (keys from Railway environment variables) ──────────────────────
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
    student_id:         str
    video_path:         str
    duration_s:         float
    total_frames:       int
    processed_frames:   int
    interviewee_side:   str  = "unknown"
    interviewee_reason: str  = ""
    transcript_summary: str  = ""
    violations:         list = field(default_factory=list)
    cheating_score:     int  = 0
    risk_level:         str  = "LOW"
    counts:             dict = field(default_factory=dict)
    timeline:           list = field(default_factory=list)
    summary:            str  = ""

    def to_dict(self):
        return {
            "student_id":           self.student_id,
            "video_path":           self.video_path,
            "duration_s":           round(self.duration_s, 2),
            "total_frames":         self.total_frames,
            "processed_frames":     self.processed_frames,
            "interviewee_side":     self.interviewee_side,
            "interviewee_reason":   self.interviewee_reason,
            "transcript_summary":   self.transcript_summary,
            "cheating_score":       self.cheating_score,
            "risk_level":           self.risk_level,
            "total_violations":     len(self.violations),
            "counts":               self.counts,
            "violations":           [v.__dict__ for v in self.violations[:100]],
            "timeline":             self.timeline,
            "summary":              self.summary,
        }


# ── Step 1: Extract audio ─────────────────────────────────────────────────────
def extract_audio(video_path: str) -> Optional[str]:
    try:
        import subprocess
        audio_path = tempfile.mktemp(suffix=".mp3")
        result = subprocess.run([
            "ffmpeg", "-i", video_path,
            "-vn", "-acodec", "libmp3lame",
            "-ar", "16000", "-ac", "1", "-q:a", "4",
            audio_path, "-y", "-loglevel", "error"
        ], capture_output=True, timeout=120)
        if result.returncode == 0 and Path(audio_path).exists():
            return audio_path
        return None
    except Exception as e:
        print(f"[WARN] Audio extraction failed: {e}")
        return None


# ── Step 2: Transcribe with Whisper ───────────────────────────────────────────
def transcribe_audio(audio_path: str) -> Optional[dict]:
    try:
        with open(audio_path, "rb") as f:
            response = openai_client.audio.transcriptions.create(
                model="whisper-1",
                file=f,
                response_format="verbose_json",
                timestamp_granularities=["segment"]
            )
        segments = []
        for seg in (response.segments or []):
            segments.append({"start": seg.start, "end": seg.end, "text": seg.text.strip()})
        return {"full_text": response.text, "segments": segments, "language": response.language}
    except Exception as e:
        print(f"[WARN] Transcription failed: {e}")
        return None


# ── Step 3: Claude identifies interviewee ─────────────────────────────────────
def identify_interviewee(transcript: dict) -> dict:
    try:
        full_text = transcript.get("full_text", "")
        if not full_text or len(full_text) < 50:
            return {"side": "LEFT", "reasoning": "Transcript too short, defaulting to LEFT"}

        prompt = f"""You are analyzing a job interview transcript recorded on Zoom/Meet gallery view.
Two people are side by side on screen.

TRANSCRIPT:
{full_text[:3000]}

The INTERVIEWER asks questions ("Tell me about yourself", "What are your strengths?")
The INTERVIEWEE answers with their experience and background.

In video calls, the local user (interviewer) is typically on the RIGHT.
The remote user (interviewee joining the call) is typically on the LEFT.

Identify which side the interviewee is on.

Respond ONLY in this exact JSON format with no other text:
{{"interviewee_side": "LEFT" or "RIGHT", "confidence": "HIGH" or "MEDIUM" or "LOW", "reasoning": "brief explanation"}}"""

        response = anthropic_client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}]
        )

        import json
        text = response.content[0].text.strip().replace("```json","").replace("```","").strip()
        parsed = json.loads(text)
        return {
            "side":      parsed.get("interviewee_side", "LEFT").upper(),
            "confidence": parsed.get("confidence", "LOW"),
            "reasoning": parsed.get("reasoning", ""),
        }
    except Exception as e:
        print(f"[WARN] Claude identification failed: {e}")
        return {"side": "LEFT", "reasoning": f"Defaulting to LEFT (error: {e})"}


# ── Head pose helper ──────────────────────────────────────────────────────────
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


# ── Main detector ─────────────────────────────────────────────────────────────
class InterviewCheatingDetector:
    def __init__(self, video_path, student_id="",
                 process_every_n_frames=3, yaw_threshold=25.0,
                 pitch_down_threshold=-15.0, head_turn_consec=8,
                 gaze_threshold=0.28, offscreen_duration_s=3.0,
                 object_conf=0.40, cooldown_s=4.0):
        self.video_path    = video_path
        self.student_id    = student_id or Path(video_path).stem
        self.process_every = process_every_n_frames
        self.yaw_thresh    = yaw_threshold
        self.pitch_down    = pitch_down_threshold
        self.head_consec   = head_turn_consec
        self.gaze_thresh   = gaze_threshold
        self.offscreen_dur = offscreen_duration_s
        self.obj_conf      = object_conf
        self.cooldown      = cooldown_s
        self._last_v: dict = {}
        self._yolo = None

    def _load_yolo(self):
        if self._yolo is None:
            try:
                from ultralytics import YOLO
                self._yolo = YOLO("yolov8n.pt")
            except Exception as e:
                print(f"[WARN] YOLO unavailable: {e}")
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
            raise ValueError(f"Cannot open video: {self.video_path}")
        fps        = cap.get(cv2.CAP_PROP_FPS) or 25
        total_f    = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_s = total_f / fps
        w          = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h          = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        result = DetectionResult(
            student_id=self.student_id, video_path=self.video_path,
            duration_s=duration_s, total_frames=total_f, processed_frames=0,
        )

        # ── Phase 1: Identify interviewee via transcript ───────────────────────
        interviewee_side   = "LEFT"
        interviewee_reason = "Default"
        transcript_summary = ""

        print(f"[1/3] Extracting audio...")
        audio_path = extract_audio(self.video_path)
        if audio_path:
            print(f"[2/3] Transcribing with Whisper...")
            transcript = transcribe_audio(audio_path)
            if transcript:
                transcript_summary = transcript["full_text"][:500]
                print(f"[3/3] Claude identifying interviewee...")
                info               = identify_interviewee(transcript)
                interviewee_side   = info.get("side", "LEFT")
                interviewee_reason = info.get("reasoning", "")
                print(f"[INFO] Interviewee on: {interviewee_side} — {interviewee_reason}")
            try: Path(audio_path).unlink()
            except: pass

        result.interviewee_side   = interviewee_side
        result.interviewee_reason = interviewee_reason
        result.transcript_summary = transcript_summary

        # Crop to interviewee side
        mid     = w // 2
        x_start = 0   if interviewee_side == "LEFT"  else mid
        x_end   = mid if interviewee_side == "LEFT"  else w

        # ── Phase 2: Frame-by-frame detection ─────────────────────────────────
        cap        = cv2.VideoCapture(self.video_path)
        violations = []
        bucket_size = 10
        n_buckets   = max(1, int(duration_s / bucket_size) + 1)
        timeline    = [{"time_s": i*bucket_size, "violations": 0} for i in range(n_buckets)]

        head_turn_count = look_down_count = 0
        offscreen_start = None
        processed       = 0
        yolo            = self._load_yolo()

        with mp_face_mesh.FaceMesh(
            static_image_mode=False, max_num_faces=3,
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

                crop   = frame[:, x_start:x_end]
                rgb    = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                cw     = crop.shape[1]
                bucket = min(int(current_time / bucket_size), n_buckets - 1)

                def add_v(vtype, detail, severity="MEDIUM"):
                    if self._can_add(vtype, current_time):
                        violations.append(Violation(frame_idx, round(current_time,2), vtype, detail, severity))
                        timeline[bucket]["violations"] += 1

                # Face count
                fd      = face_det.process(rgb)
                n_faces = len(fd.detections) if fd.detections else 0
                if n_faces == 0:
                    add_v("interviewee_absent", "Interviewee left screen", "MEDIUM")
                if n_faces > 1:
                    add_v("third_person_detected", f"{n_faces} faces on interviewee side", "HIGH")

                # Face mesh
                mesh = face_mesh.process(rgb)
                if mesh.multi_face_landmarks:
                    lm = mesh.multi_face_landmarks[0].landmark

                    # Head pose
                    try:
                        pitch, yaw, roll = get_head_pose(lm, cw, h)
                        if abs(yaw) > self.yaw_thresh:
                            head_turn_count += 1
                            if head_turn_count >= self.head_consec:
                                d = "left" if yaw < 0 else "right"
                                add_v("head_turn", f"Head turned {d} ({abs(yaw):.0f}°)", "MEDIUM")
                                head_turn_count = 0
                        else:
                            head_turn_count = 0

                        if pitch < self.pitch_down:
                            look_down_count += 1
                            if look_down_count >= self.head_consec:
                                add_v("looking_down", f"Head down ({pitch:.0f}°) — reading notes", "MEDIUM")
                                look_down_count = 0
                        else:
                            look_down_count = 0
                    except: pass

                    # Gaze
                    try:
                        lg  = get_iris_gaze(lm, LEFT_IRIS,  LEFT_EAR_H,  cw, h)
                        rg  = get_iris_gaze(lm, RIGHT_IRIS, RIGHT_EAR_H, cw, h)
                        dev = abs((lg+rg)/2 - 0.5)
                        if dev > self.gaze_thresh:
                            if offscreen_start is None: offscreen_start = current_time
                            elif current_time - offscreen_start >= self.offscreen_dur:
                                d = "left" if (lg+rg)/2 < 0.5 else "right"
                                add_v("sustained_gaze_off", f"Eyes looking {d} for {self.offscreen_dur:.0f}+s", "MEDIUM")
                                offscreen_start = None
                        else:
                            offscreen_start = None
                    except: pass

                # Objects
                if yolo and frame_idx % 12 == 0:
                    try:
                        for r in yolo(rgb, verbose=False, conf=self.obj_conf):
                            for box in r.boxes:
                                cls = yolo.model.names[int(box.cls)].lower()
                                if "phone" in cls:
                                    add_v("phone_detected", "Phone visible — googling answers", "HIGH")
                                elif any(x in cls for x in ["book","laptop","tablet","earphone","airpod","headphone"]):
                                    add_v("prohibited_object", f"Detected: {cls}", "MEDIUM")
                    except: pass

        cap.release()

        counts = {
            "third_person_detected": sum(1 for v in violations if v.type=="third_person_detected"),
            "phone_detected":        sum(1 for v in violations if v.type=="phone_detected"),
            "prohibited_object":     sum(1 for v in violations if v.type=="prohibited_object"),
            "head_turn":             sum(1 for v in violations if v.type=="head_turn"),
            "looking_down":          sum(1 for v in violations if v.type=="looking_down"),
            "sustained_gaze_off":    sum(1 for v in violations if v.type=="sustained_gaze_off"),
            "interviewee_absent":    sum(1 for v in violations if v.type=="interviewee_absent"),
        }
        raw = (
            counts["third_person_detected"]*25 + counts["phone_detected"]*22 +
            counts["prohibited_object"]*15     + counts["sustained_gaze_off"]*10 +
            counts["head_turn"]*8              + counts["looking_down"]*8 +
            counts["interviewee_absent"]*5
        )
        score = min(100, raw)
        risk  = "HIGH" if score >= 60 else "MEDIUM" if score >= 25 else "LOW"

        parts = []
        if counts["third_person_detected"]>0: parts.append(f"3rd person {counts['third_person_detected']}x")
        if counts["phone_detected"]>0:        parts.append(f"Phone {counts['phone_detected']}x")
        if counts["prohibited_object"]>0:     parts.append(f"Notes/book {counts['prohibited_object']}x")
        if counts["sustained_gaze_off"]>0:    parts.append(f"Off-screen gaze {counts['sustained_gaze_off']}x")
        if counts["head_turn"]>0:             parts.append(f"Head turned {counts['head_turn']}x")
        if counts["looking_down"]>0:          parts.append(f"Looked down {counts['looking_down']}x")
        summary = ", ".join(parts) if parts else "No suspicious activity detected"

        result.violations=violations; result.processed_frames=processed
        result.cheating_score=score;  result.risk_level=risk
        result.counts=counts;         result.timeline=timeline
        result.summary=summary
        return result
