# 4ml_advanced_concentration_tracker.py
import cv2
import mediapipe as mp
import numpy as np
import sounddevice as sd
import time
import threading
from collections import deque
import math

# ---------------- ENHANCED CONFIG ----------------
CALIB_SECONDS = 3.0
EAR_BLINK_THRESHOLD = 0.18
EAR_SMOOTHING = 5
BLINK_CONSEC_FRAMES = 2
FACE_DET_CONF = 0.45
EYE_VARIANCE_THRESHOLD = 200.0
EYE_MEAN_DARK = 45.0
GAZE_X_DELTA = 0.07
GAZE_Y_DELTA = 0.06
SCORE_SMOOTH = 8
NOISE_SENSITIVITY = 2.0
AUDIO_CALIB_SECONDS = 1.0
AUDIO_SR = 22050
AUDIO_BLOCKSIZE = 1024
EYES_CLOSED_SECONDS = 3.0
NO_FACE_SECONDS = 10.0

# NEW: Advanced settings
HEAD_MOVEMENT_THRESHOLD = 0.15  # Head pose deviation threshold
MICRO_EXPRESSION_WINDOW = 10    # Frames to track subtle movements
CONCENTRATION_HISTORY_SIZE = 60 # For trend analysis
FATIGUE_ALERT_MINUTES = 2.0     # Time for fatigue warning
GAZE_STABILITY_THRESHOLD = 0.02 # Iris position stability
# ----------------------------------------

# MediaPipe
mp_face_mesh = mp.solutions.face_mesh
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

face_mesh = mp_face_mesh.FaceMesh(
    refine_landmarks=True, 
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.4)

# Enhanced landmark indices
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
LEFT_IRIS = 468
RIGHT_IRIS = 473
FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 
             397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 
             172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]

# Head pose estimation points
NOSE_TIP = 1
CHIN = 199
LEFT_EYE_CORNER = 33
RIGHT_EYE_CORNER = 263
LEFT_MOUTH = 61
RIGHT_MOUTH = 291

# audio globals (thread-safe)
_audio_rms = 0.0
_audio_lock = threading.Lock()
_audio_stream = None
_audio_baseline = 1e-6

def audio_callback(indata, frames, time_info, status):
    global _audio_rms
    mono = np.mean(indata, axis=1) if indata.ndim > 1 else indata[:,0]
    rms = float(np.sqrt(np.mean(np.square(mono))))
    with _audio_lock:
        _audio_rms = rms

def start_audio():
    global _audio_stream
    try:
        _audio_stream = sd.InputStream(callback=audio_callback,
                                       blocksize=AUDIO_BLOCKSIZE,
                                       samplerate=AUDIO_SR,
                                       channels=1)
        _audio_stream.start()
        return True
    except Exception as e:
        print("Audio stream start failed:", e)
        return False

def calibrate_audio_baseline():
    global _audio_baseline
    try:
        print(f"Calibrating microphone for {AUDIO_CALIB_SECONDS:.1f}s — please be quiet...")
        rec = sd.rec(int(AUDIO_CALIB_SECONDS * AUDIO_SR), samplerate=AUDIO_SR, channels=1, dtype='float64')
        sd.wait()
        mono = rec[:,0]
        _audio_baseline = max(1e-6, float(np.sqrt(np.mean(np.square(mono)))))
        print(f"Audio baseline RMS = {_audio_baseline:.6f}")
    except Exception as e:
        print("Audio calibration failed:", e)
        _audio_baseline = 1e-6

# ---------- ENHANCED Helpers ----------
def eye_aspect_ratio(landmarks, eye_indices, w, h):
    try:
        pts = [(landmarks[i].x * w, landmarks[i].y * h) for i in eye_indices]
        A = np.linalg.norm(np.array(pts[1]) - np.array(pts[5]))
        B = np.linalg.norm(np.array(pts[2]) - np.array(pts[4]))
        C = np.linalg.norm(np.array(pts[0]) - np.array(pts[3]))
        if C <= 1e-6:
            return 0.0
        return (A + B) / (2.0 * C)
    except Exception:
        return 0.0

def eye_region_stats(gray, landmarks, eye_indices, w, h, pad=6):
    xs = [int(landmarks[i].x * w) for i in eye_indices]
    ys = [int(landmarks[i].y * h) for i in eye_indices]
    x1 = max(min(xs) - pad, 0); x2 = min(max(xs) + pad, w-1)
    y1 = max(min(ys) - pad, 0); y2 = min(max(ys) + pad, h-1)
    if x2 <= x1 or y2 <= y1:
        return None
    region = gray[y1:y2, x1:x2]
    if region.size == 0:
        return None
    return float(np.mean(region)), float(np.var(region))

def get_iris_avg(landmarks):
    try:
        return (landmarks[LEFT_IRIS].x + landmarks[RIGHT_IRIS].x) / 2.0, \
               (landmarks[LEFT_IRIS].y + landmarks[RIGHT_IRIS].y) / 2.0
    except Exception:
        return None, None

# NEW: Head pose estimation
def estimate_head_pose(landmarks, w, h):
    """Estimate head pose using simple geometric approach"""
    try:
        # Convert key points to image coordinates
        nose = [landmarks[NOSE_TIP].x * w, landmarks[NOSE_TIP].y * h]
        chin = [landmarks[CHIN].x * w, landmarks[CHIN].y * h]
        left_eye = [landmarks[LEFT_EYE_CORNER].x * w, landmarks[LEFT_EYE_CORNER].y * h]
        right_eye = [landmarks[RIGHT_EYE_CORNER].x * w, landmarks[RIGHT_EYE_CORNER].y * h]
        
        # Calculate face center and dimensions
        face_center_x = (left_eye[0] + right_eye[0]) / 2
        face_center_y = (left_eye[1] + right_eye[1]) / 2
        
        # Simple pitch estimation (vertical head tilt)
        vertical_ratio = abs(nose[1] - chin[1]) / abs(left_eye[1] - chin[1]) if abs(left_eye[1] - chin[1]) > 0 else 1.0
        
        # Simple yaw estimation (horizontal head turn)
        eye_distance = abs(left_eye[0] - right_eye[0])
        nose_offset = abs(nose[0] - face_center_x)
        horizontal_ratio = nose_offset / (eye_distance / 2) if eye_distance > 0 else 0.0
        
        return vertical_ratio, horizontal_ratio
    except Exception:
        return 1.0, 0.0

# NEW: Micro-expression detector
class MicroExpressionDetector:
    def __init__(self, window_size=10):
        self.window_size = window_size
        self.expression_history = deque(maxlen=window_size)
        self.mouth_history = deque(maxlen=window_size)
        
    def detect_micro_expressions(self, landmarks):
        """Detect subtle facial movements that indicate distraction"""
        try:
            # Mouth openness variation
            upper_lip = landmarks[13].y
            lower_lip = landmarks[14].y
            mouth_openness = abs(upper_lip - lower_lip)
            
            # Eyebrow raise detection
            left_brow = landmarks[65].y
            right_brow = landmarks[295].y
            
            self.expression_history.append((mouth_openness, left_brow, right_brow))
            
            if len(self.expression_history) < self.window_size:
                return False
                
            # Calculate variation in recent frames
            mouth_vars = [item[0] for item in self.expression_history]
            brow_vars = [(item[1] + item[2])/2 for item in self.expression_history]
            
            mouth_std = np.std(mouth_vars)
            brow_std = np.std(brow_vars)
            
            # High variation indicates micro-expressions
            return mouth_std > 0.005 or brow_std > 0.004
            
        except Exception:
            return False

# NEW: Gaze stability analyzer
class GazeStabilityAnalyzer:
    def __init__(self, window_size=15):
        self.window_size = window_size
        self.gaze_history = deque(maxlen=window_size)
        
    def add_gaze_point(self, x, y):
        if x is not None and y is not None:
            self.gaze_history.append((x, y))
            
    def get_stability(self):
        if len(self.gaze_history) < 2:
            return 1.0
            
        # Calculate movement variance
        movements = []
        for i in range(1, len(self.gaze_history)):
            dx = self.gaze_history[i][0] - self.gaze_history[i-1][0]
            dy = self.gaze_history[i][1] - self.gaze_history[i-1][1]
            movements.append(math.sqrt(dx*dx + dy*dy))
            
        if not movements:
            return 1.0
            
        avg_movement = np.mean(movements)
        stability = max(0, 1 - (avg_movement / GAZE_STABILITY_THRESHOLD))
        return stability

# NEW: Advanced concentration calculator
def compute_advanced_concentration(gaze_ok, head_ok, blink_recent, occluded, noise_flag, 
                                 head_stability, gaze_stability, micro_expression_detected,
                                 time_focused):
    """
    Enhanced concentration scoring with multiple factors
    """
    # Base factors (50%)
    gaze_score = 1.0 if gaze_ok else 0.0
    head_score = 1.0 if head_ok else 0.0
    base_score = 0.3 * gaze_score + 0.2 * head_score
    
    # Stability factors (30%)
    stability_score = 0.15 * head_stability + 0.15 * gaze_stability
    
    # Behavioral factors (20%)
    behavioral_score = 0.1
    if not blink_recent:
        behavioral_score += 0.05
    if not micro_expression_detected:
        behavioral_score += 0.05
        
    # Penalties
    penalties = 0.0
    if occluded:
        penalties += 0.3
    if noise_flag:
        penalties += 0.1
    if blink_recent:
        penalties += 0.1
        
    # Time bonus (up to 10%)
    time_bonus = min(0.1, time_focused / 300.0)  # Max bonus after 5 minutes
    
    total_score = base_score + stability_score + behavioral_score + time_bonus - penalties
    return int(np.clip(total_score * 100.0, 0, 100))

# NEW: Fatigue detector
class FatigueDetector:
    def __init__(self, alert_minutes=2.0):
        self.alert_minutes = alert_minutes
        self.start_time = time.time()
        self.blink_count = 0
        self.last_blink_time = time.time()
        
    def add_blink(self):
        self.blink_count += 1
        self.last_blink_time = time.time()
        
    def get_fatigue_level(self):
        current_time = time.time()
        session_duration = (current_time - self.start_time) / 60.0  # minutes
        
        if session_duration < self.alert_minutes:
            return 0
            
        # Calculate blink rate (blinks per minute)
        recent_blinks = self.blink_count
        blink_rate = recent_blinks / session_duration
        
        # High blink rate or long session indicates fatigue
        if blink_rate > 25 or session_duration > 15:
            return 2  # High fatigue
        elif blink_rate > 15 or session_duration > 8:
            return 1  # Moderate fatigue
            
        return 0

# ---------- Start audio thread and calibrate ----------
audio_ok = start_audio()
if audio_ok:
    calibrate_audio_baseline()
else:
    print("Warning: audio disabled; noise detection will be off.")

# ---------- Camera and calibration ----------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Cannot open camera. Close other apps or check device.")
    raise SystemExit

cv2.namedWindow("Advanced Concentration Tracker", cv2.WINDOW_NORMAL)
print("Camera opened. Starting gaze calibration — look straight at the camera now.")

calib_x = []
calib_y = []
calib_start = time.time()
while time.time() - calib_start < CALIB_SECONDS:
    ret, frame = cap.read()
    if not ret:
        continue
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    det = face_detection.process(rgb)
    mesh = face_mesh.process(rgb)
    if mesh.multi_face_landmarks and det.detections:
        landmarks = mesh.multi_face_landmarks[0].landmark
        avgx, avgy = get_iris_avg(landmarks)
        if avgx is not None:
            calib_x.append(avgx); calib_y.append(avgy)
    cv2.putText(frame, "Calibrating gaze (keep eyes on camera)...", (20,40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
    cv2.imshow("Advanced Concentration Tracker", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        raise SystemExit

if not calib_x:
    print("Calibration failed: no face/iris detected. Retry with better lighting/position.")
    cap.release()
    cv2.destroyAllWindows()
    raise SystemExit

baseline_x = float(np.mean(calib_x))
baseline_y = float(np.mean(calib_y))
print(f"Calibration complete. Baseline iris center = ({baseline_x:.3f}, {baseline_y:.3f})")

# ---------- Initialize advanced components ----------
score_buf = deque(maxlen=SCORE_SMOOTH)
ear_buf = deque(maxlen=EAR_SMOOTHING)
frame_no = 0
blink_frames = 0
last_blink_time = 0.0
BLINK_MIN_SEP = 0.35
concentration_start_time = time.time()
last_good_concentration_time = time.time()

# Advanced detectors
micro_detector = MicroExpressionDetector(MICRO_EXPRESSION_WINDOW)
gaze_stability = GazeStabilityAnalyzer()
fatigue_detector = FatigueDetector(FATIGUE_ALERT_MINUTES)
concentration_history = deque(maxlen=CONCENTRATION_HISTORY_SIZE)

# Timers
eyes_closed_start = None
no_face_start = None

print("Advanced tracker running. Press 'q' in the window to quit.")

# ---------- Enhanced main loop ----------
while True:
    ret, frame = cap.read()
    if not ret:
        print("Frame grab failed; exiting main loop.")
        break

    frame_no += 1
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Face detection confidence and mesh
    det = face_detection.process(rgb)
    mesh = face_mesh.process(rgb)
    face_conf = 0.0
    if det.detections:
        face_conf = max([d.score[0] for d in det.detections])

    occluded = False
    blink_event = False
    gaze_dir = "UNKNOWN"
    concentration = 0
    head_stability = 1.0
    gaze_stability_score = 1.0
    micro_expression_detected = False
    head_pose_ok = True

    if mesh.multi_face_landmarks and face_conf >= FACE_DET_CONF:
        # Reset no-face timer since we have a face now
        no_face_start = None

        lm = mesh.multi_face_landmarks[0].landmark

        # Draw enhanced mesh
        mp_drawing.draw_landmarks(frame, mesh.multi_face_landmarks[0], mp_face_mesh.FACEMESH_TESSELATION,
                                  mp_drawing.DrawingSpec(color=(0,200,0), thickness=1, circle_radius=1),
                                  mp_drawing.DrawingSpec(color=(0,150,255), thickness=1))

        # Draw face oval for better visualization
        face_oval_points = [(int(lm[i].x * w), int(lm[i].y * h)) for i in FACE_OVAL]
        for i in range(len(face_oval_points)):
            cv2.line(frame, face_oval_points[i], face_oval_points[(i+1) % len(face_oval_points)], 
                    (0, 255, 0), 1)

        # Enhanced EAR with smoothing
        left_ear = eye_aspect_ratio(lm, LEFT_EYE, w, h)
        right_ear = eye_aspect_ratio(lm, RIGHT_EYE, w, h)
        avg_ear = (left_ear + right_ear) / 2.0
        ear_buf.append(avg_ear)
        smoothed_ear = np.mean(ear_buf) if ear_buf else avg_ear

        if smoothed_ear > 0 and smoothed_ear < EAR_BLINK_THRESHOLD:
            blink_frames += 1
        else:
            if blink_frames >= BLINK_CONSEC_FRAMES:
                now = time.time()
                if now - last_blink_time > BLINK_MIN_SEP:
                    blink_event = True
                    last_blink_time = now
                    fatigue_detector.add_blink()
            blink_frames = 0

        # Eyes closed timer
        if smoothed_ear > 0 and smoothed_ear < EAR_BLINK_THRESHOLD:
            if eyes_closed_start is None:
                eyes_closed_start = time.time()
            if time.time() - eyes_closed_start >= EYES_CLOSED_SECONDS:
                cv2.putText(frame, "EYES CLOSED > 3s", (20, 300),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        else:
            eyes_closed_start = None

        # Occlusion detection
        left_stats = eye_region_stats(gray, lm, LEFT_EYE, w, h)
        right_stats = eye_region_stats(gray, lm, RIGHT_EYE, w, h)
        if left_stats is None or right_stats is None:
            occluded = True
        else:
            lmean, lvar = left_stats; rmean, rvar = right_stats
            if (lvar < EYE_VARIANCE_THRESHOLD or lmean < EYE_MEAN_DARK) and \
               (rvar < EYE_VARIANCE_THRESHOLD or rmean < EYE_MEAN_DARK):
                occluded = True

        # Head pose estimation
        vertical_ratio, horizontal_ratio = estimate_head_pose(lm, w, h)
        head_pose_ok = (vertical_ratio < 1.3 and horizontal_ratio < 0.8)
        head_stability = 1.0 - min(1.0, horizontal_ratio * 1.5)

        # Gaze direction with stability analysis
        avgx, avgy = get_iris_avg(lm)
        gaze_stability.add_gaze_point(avgx, avgy)
        gaze_stability_score = gaze_stability.get_stability()
        
        if avgx is None:
            gaze_dir = "UNKNOWN"
        else:
            dx = avgx - baseline_x
            dy = avgy - baseline_y
            if abs(dx) <= GAZE_X_DELTA and abs(dy) <= GAZE_Y_DELTA:
                gaze_dir = "CENTER"
            elif abs(dx) > abs(dy):
                gaze_dir = "LEFT" if dx < 0 else "RIGHT"
            else:
                gaze_dir = "UP" if dy < 0 else "DOWN"

        # Micro-expression detection
        micro_expression_detected = micro_detector.detect_micro_expressions(lm)

        # Audio noise check
        with _audio_lock:
            current_rms = _audio_rms
        noise_flag = False
        if _audio_baseline > 0 and current_rms > _audio_baseline * NOISE_SENSITIVITY:
            noise_flag = True

        # Time tracking for concentration
        current_time = time.time()
        time_focused = current_time - concentration_start_time

        # Compute advanced concentration score
        gaze_ok = (gaze_dir == "CENTER")
        head_ok = head_pose_ok
        concentration = compute_advanced_concentration(
            gaze_ok, head_ok, blink_event, occluded, noise_flag,
            head_stability, gaze_stability_score, micro_expression_detected,
            time_focused
        )
        
        # Update concentration history for trend analysis
        concentration_history.append(concentration)
        
    else:
        # No reliable face
        concentration = 0
        occluded = True
        gaze_dir = "NO_FACE"
        noise_flag = False
        concentration_start_time = time.time()  # Reset focus timer

        # No-face timer
        if no_face_start is None:
            no_face_start = time.time()
        else:
            elapsed_no_face = time.time() - no_face_start
            cv2.putText(frame, f"No face: {int(elapsed_no_face)}s/{int(NO_FACE_SECONDS)}s", (20, 300),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if elapsed_no_face >= NO_FACE_SECONDS:
                print(f"No face detected for {NO_FACE_SECONDS} seconds. Exiting...")
                break

    # Update concentration buffer and calculate smooth score
    score_buf.append(concentration)
    smooth_score = int(np.mean(score_buf)) if len(score_buf) > 0 else concentration

    # Enhanced status with fatigue detection
    fatigue_level = fatigue_detector.get_fatigue_level()
    
    if not mesh.multi_face_landmarks or face_conf < FACE_DET_CONF:
        status = "NO FACE"
    elif occluded:
        status = "OCCLUDED"
    elif fatigue_level == 2:
        status = "FATIGUED!"
    elif fatigue_level == 1:
        status = "TIRED"
    else:
        with _audio_lock:
            curr = _audio_rms
        noisy = (_audio_baseline > 0 and curr > _audio_baseline * NOISE_SENSITIVITY)
        if noisy:
            status = "NOISY"
        elif blink_event:
            status = "BLINK"
        elif micro_expression_detected:
            status = "DISTRACTED"
        elif smooth_score < 55:
            status = "DISTRACTED"
        else:
            status = "CONCENTRATED"

    # Enhanced UI with more information
    bar_x, bar_y, bar_w, bar_h = 18, 22, 320, 30
    fill = int((smooth_score/100.0) * bar_w)
    
    # Dynamic color based on score and fatigue
    if fatigue_level > 0:
        color = (0, 100, 255)  # Orange for fatigue
    elif smooth_score >= 75:
        color = (0, 200, 0)    # Green for high concentration
    elif smooth_score >= 50:
        color = (0, 140, 255)  # Yellow for medium
    else:
        color = (0, 80, 200)   # Red for low
        
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x+bar_w, bar_y+bar_h), (30,30,30), -1)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x+fill, bar_y+bar_h), color, -1)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x+bar_w, bar_y+bar_h), (200,200,200), 2)
    cv2.putText(frame, f"{smooth_score}%", (bar_x+bar_w+10, bar_y+22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    # Enhanced information display
    y_offset = 70
    cv2.putText(frame, f"Status: {status}", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
    y_offset += 35
    cv2.putText(frame, f"Gaze: {gaze_dir}", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    y_offset += 25
    cv2.putText(frame, f"Stability: {gaze_stability_score:.2f}", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)
    y_offset += 20
    cv2.putText(frame, f"Head Pose: {'OK' if head_pose_ok else 'MOVING'}", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)
    
    if blink_event:
        y_offset += 25
        cv2.putText(frame, "BLINK DETECTED", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
    if noise_flag:
        y_offset += 25
        cv2.putText(frame, "BACKGROUND NOISE", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
    if micro_expression_detected:
        y_offset += 25
        cv2.putText(frame, "MICRO-EXPRESSIONS", (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,165,255), 2)
    if fatigue_level > 0:
        y_offset += 25
        fatigue_text = "TAKE A BREAK!" if fatigue_level == 2 else "You seem tired"
        cv2.putText(frame, fatigue_text, (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,100,255), 2)

    # Session info
    cv2.putText(frame, f"Frame: {frame_no}", (w-150, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)
    
    # Trend indicator (simple)
    if len(concentration_history) > 10:
        recent_avg = np.mean(list(concentration_history)[-10:])
        previous_avg = np.mean(list(concentration_history)[-20:-10]) if len(concentration_history) >= 20 else recent_avg
        trend = "↑" if recent_avg > previous_avg + 5 else "↓" if recent_avg < previous_avg - 5 else "→"
        cv2.putText(frame, f"Trend: {trend}", (w-150, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)

    cv2.imshow("Advanced Concentration Tracker", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        # Reset calibration on 'r' key
        print("Resetting calibration...")
        calib_x.clear()
        calib_y.clear()
        calib_start = time.time()
        while time.time() - calib_start < CALIB_SECONDS:
            ret, cal_frame = cap.read()
            if not ret:
                continue
            cal_frame = cv2.flip(cal_frame, 1)
            rgb_cal = cv2.cvtColor(cal_frame, cv2.COLOR_BGR2RGB)
            det_cal = face_detection.process(rgb_cal)
            mesh_cal = face_mesh.process(rgb_cal)
            if mesh_cal.multi_face_landmarks and det_cal.detections:
                landmarks_cal = mesh_cal.multi_face_landmarks[0].landmark
                avgx, avgy = get_iris_avg(landmarks_cal)
                if avgx is not None:
                    calib_x.append(avgx); calib_y.append(avgy)
            cv2.putText(cal_frame, "Recalibrating... Look straight", (20,40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
            cv2.imshow("Advanced Concentration Tracker", cal_frame)
            cv2.waitKey(1)
        
        if calib_x:
            baseline_x = float(np.mean(calib_x))
            baseline_y = float(np.mean(calib_y))
            print(f"Recalibration complete. New baseline = ({baseline_x:.3f}, {baseline_y:.3f})")
        else:
            print("Recalibration failed. Using previous baseline.")

# Cleanup
try:
    if _audio_stream is not None:
        _audio_stream.stop()
        _audio_stream.close()
except Exception:
    pass

cap.release()
cv2.destroyAllWindows()

# Session summary
if concentration_history:
    avg_concentration = np.mean(concentration_history)
    max_concentration = np.max(concentration_history)
    print(f"\n--- Session Summary ---")
    print(f"Average Concentration: {avg_concentration:.1f}%")
    print(f"Peak Concentration: {max_concentration}%")
    print(f"Total Blinks: {fatigue_detector.blink_count}")
    print(f"Session Duration: {(time.time() - fatigue_detector.start_time)/60:.1f} minutes")

print("Exited cleanly.")