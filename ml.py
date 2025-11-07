# fixed_concentration_tracker.py
import cv2
import numpy as np
import time
from collections import deque

print("Advanced Concentration Tracker - Fixed Blink Detection")

# Configuration
CALIB_SECONDS = 3.0
SCORE_SMOOTH = 8
NO_FACE_SECONDS = 10.0
EYES_CLOSED_SECONDS = 3.0

# Blink detection settings
BLINK_RATIO_THRESHOLD = 4.0  # Higher = more sensitive to blinks
BLINK_FRAMES_THRESHOLD = 3   # Consecutive frames to register blink
BLINK_COOLDOWN = 0.3         # Seconds between blink detection

# Load detectors
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Initialize camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("ERROR: Cannot open camera")
    exit()

cv2.namedWindow("Concentration Tracker", cv2.WINDOW_NORMAL)

# Calibration phase
print("Calibrating - look straight at camera...")
calib_positions = []
calib_start = time.time()

while time.time() - calib_start < CALIB_SECONDS:
    ret, frame = cap.read()
    if not ret:
        continue
        
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    if len(faces) > 0:
        x, y, w, h = faces[0]
        face_center = (x + w/2, y + h/2)
        calib_positions.append(face_center)
        
    cv2.putText(frame, "Calibrating... Look straight ahead", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.imshow("Concentration Tracker", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        exit()

if not calib_positions:
    print("Calibration failed - no face detected")
    cap.release()
    cv2.destroyAllWindows()
    exit()

# Calculate baseline position
calib_array = np.array(calib_positions)
baseline_x, baseline_y = np.mean(calib_array, axis=0)
print(f"Calibration complete. Baseline: ({baseline_x:.1f}, {baseline_y:.1f})")

# Blink detection variables
blink_counter = 0
eyes_closed_start = None
no_face_start = None
frame_count = 0
consecutive_eyes_closed = 0
last_blink_time = 0
eye_ratio_history = deque(maxlen=5)  # Store recent eye ratios

score_buf = deque(maxlen=SCORE_SMOOTH)

print("Tracking started. Press 'q' to quit, 'r' to recalibrate")

def calculate_eye_ratio(eye_region):
    """Calculate eye aspect ratio for blink detection"""
    if eye_region.size == 0:
        return 0
    
    height, width = eye_region.shape
    
    # For blink detection, we use aspect ratio
    if height == 0 or width == 0:
        return 0
    
    # Simple aspect ratio (width/height)
    ratio = width / height
    
    return ratio

def get_eye_status(eyes, face_roi):
    """Determine if eyes are open or closed"""
    if len(eyes) < 2:
        return "NO_EYES", 0
    
    total_ratio = 0
    valid_eyes = 0
    
    for (ex, ey, ew, eh) in eyes:
        eye_region = face_roi[ey:ey+eh, ex:ex+ew]
        ratio = calculate_eye_ratio(eye_region)
        
        if ratio > 0:  # Valid eye region
            total_ratio += ratio
            valid_eyes += 1
    
    if valid_eyes == 0:
        return "NO_EYES", 0
    
    avg_ratio = total_ratio / valid_eyes
    eye_ratio_history.append(avg_ratio)
    
    # Use historical data for better detection
    if len(eye_ratio_history) >= 3:
        recent_avg = np.mean(list(eye_ratio_history)[-3:])
    else:
        recent_avg = avg_ratio
    
    # Lower ratio = more closed eyes (since closed eyes are taller relative to width)
    if recent_avg < BLINK_RATIO_THRESHOLD:
        return "CLOSED", recent_avg
    else:
        return "OPEN", recent_ratio

def get_gaze_direction(eyes, face_center, frame_width):
    """Determine gaze direction based on eye positions"""
    if len(eyes) < 2:
        return "UNKNOWN"
    
    # Sort eyes by x position
    eyes_sorted = sorted(eyes, key=lambda e: e[0] + e[2]/2)
    left_eye_center = eyes_sorted[0][0] + eyes_sorted[0][2]/2
    right_eye_center = eyes_sorted[1][0] + eyes_sorted[1][2]/2
    
    avg_eye_x = (left_eye_center + right_eye_center) / 2
    deviation = avg_eye_x - face_center[0]
    
    threshold = frame_width * 0.05  # 5% of frame width
    
    if deviation < -threshold:
        return "LEFT"
    elif deviation > threshold:
        return "RIGHT"
    else:
        return "CENTER"

while True:
    ret, frame = cap.read()
    if not ret:
        break
        
    frame_count += 1
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    height, width = frame.shape[:2]
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    concentration = 0
    status = "NO FACE"
    gaze_dir = "UNKNOWN"
    blink_detected = False
    eye_status = "UNKNOWN"
    current_eye_ratio = 0
    
    if len(faces) > 0:
        no_face_start = None  # Reset no-face timer
        
        x, y, w, h = faces[0]
        face_center = (x + w/2, y + h/2)
        
        # Draw face bounding box
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Detect eyes within face region
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 6, minSize=(20, 20))
        
        # Eye status and blink detection
        eye_status, current_eye_ratio = get_eye_status(eyes, roi_gray)
        current_time = time.time()
        
        # Draw eye bounding boxes and info
        for (ex, ey, ew, eh) in eyes:
            eye_color = (0, 0, 255) if eye_status == "CLOSED" else (255, 0, 0)
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), eye_color, 2)
            
            # Draw eye ratio on each eye
            eye_region = roi_gray[ey:ey+eh, ex:ex+ew]
            ratio = calculate_eye_ratio(eye_region)
            cv2.putText(roi_color, f"{ratio:.1f}", (ex, ey-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # BLINK DETECTION LOGIC - FIXED
        if eye_status == "CLOSED":
            consecutive_eyes_closed += 1
            
            # Check if this is a new blink (not already counting)
            if consecutive_eyes_closed >= BLINK_FRAMES_THRESHOLD:
                # Only register blink if enough time has passed since last blink
                if current_time - last_blink_time > BLINK_COOLDOWN:
                    blink_detected = True
                    blink_counter += 1
                    last_blink_time = current_time
                    consecutive_eyes_closed = 0  # Reset counter after blink
        else:
            consecutive_eyes_closed = 0
        
        # Eyes closed timer (for long closures)
        if eye_status == "CLOSED":
            if eyes_closed_start is None:
                eyes_closed_start = current_time
            elif current_time - eyes_closed_start >= EYES_CLOSED_SECONDS:
                cv2.putText(frame, "EYES CLOSED > 3s", (width//2 - 100, height - 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        else:
            eyes_closed_start = None
        
        # Gaze direction
        gaze_dir = get_gaze_direction(eyes, face_center, width)
        
        # Calculate concentration score
        position_score = 0
        gaze_score = 0
        eye_score = 0
        
        # Position score (how centered the face is)
        center_deviation_x = abs(face_center[0] - width/2) / (width/2)
        center_deviation_y = abs(face_center[1] - height/2) / (height/2)
        position_score = 100 - (center_deviation_x * 60 + center_deviation_y * 40)
        
        # Gaze score
        gaze_score = 80 if gaze_dir == "CENTER" else 40
        
        # Eye score (penalize closed eyes)
        eye_score = 80 if eye_status == "OPEN" else 20
        
        # Combine scores
        concentration = int(0.4 * position_score + 0.4 * gaze_score + 0.2 * eye_score)
        concentration = max(0, min(100, concentration))
        
        # Determine status
        if blink_detected:
            status = "BLINK DETECTED!"
        elif eye_status == "CLOSED":
            status = "EYES CLOSED"
        elif gaze_dir != "CENTER":
            status = "LOOKING AWAY"
        elif concentration >= 70:
            status = "CONCENTRATED"
        elif concentration >= 40:
            status = "MODERATE"
        else:
            status = "DISTRACTED"
            
    else:
        # No face detected
        concentration = 0
        eye_status = "NO FACE"
        if no_face_start is None:
            no_face_start = time.time()
        else:
            elapsed = time.time() - no_face_start
            cv2.putText(frame, f"No face: {int(elapsed)}s/{int(NO_FACE_SECONDS)}s", 
                       (20, height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if elapsed >= NO_FACE_SECONDS:
                print("No face detected for too long. Exiting...")
                break

    # Smooth concentration score
    score_buf.append(concentration)
    smooth_score = int(np.mean(score_buf)) if score_buf else concentration

    # UI Display
    y_offset = 60
    
    # Concentration bar
    bar_width = 300
    fill = int((smooth_score/100) * bar_width)
    color = (0, 200, 0) if smooth_score >= 70 else (0, 140, 255) if smooth_score >= 40 else (0, 80, 200)
    
    cv2.rectangle(frame, (20, 20), (20 + bar_width, 50), (50, 50, 50), -1)
    cv2.rectangle(frame, (20, 20), (20 + fill, 50), color, -1)
    cv2.rectangle(frame, (20, 20), (20 + bar_width, 50), (200, 200, 200), 2)
    cv2.putText(frame, f"{smooth_score}%", (20 + bar_width + 10, 45), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Status information
    cv2.putText(frame, f"Status: {status}", (20, y_offset), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    y_offset += 35
    
    cv2.putText(frame, f"Gaze: {gaze_dir}", (20, y_offset), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    y_offset += 30
    
    cv2.putText(frame, f"Eyes: {eye_status} (ratio: {current_eye_ratio:.1f})", (20, y_offset), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    y_offset += 30
    
    cv2.putText(frame, f"Blinks: {blink_counter}", (20, y_offset), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    y_offset += 30
    
    # Blink detection info
    cv2.putText(frame, f"Closed frames: {consecutive_eyes_closed}/{BLINK_FRAMES_THRESHOLD}", 
                (20, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    y_offset += 25
    
    # Visual blink indicator
    if blink_detected:
        cv2.putText(frame, ">>> BLINK! <<<", (width//2 - 80, 100), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
    
    # Frame counter
    cv2.putText(frame, f"Frame: {frame_count}", (width - 150, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    # Instructions
    cv2.putText(frame, "Press 'q' to quit, 'r' to recalibrate", (20, height - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    cv2.imshow("Concentration Tracker", frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        # Recalibration
        print("Recalibrating...")
        calib_positions = []
        calib_start = time.time()
        
        while time.time() - calib_start < CALIB_SECONDS:
            ret, cal_frame = cap.read()
            if not ret:
                continue
                
            cal_frame = cv2.flip(cal_frame, 1)
            cal_gray = cv2.cvtColor(cal_frame, cv2.COLOR_BGR2GRAY)
            cal_faces = face_cascade.detectMultiScale(cal_gray, 1.3, 5)
            
            if len(cal_faces) > 0:
                x, y, w, h = cal_faces[0]
                face_center = (x + w/2, y + h/2)
                calib_positions.append(face_center)
                
            cv2.putText(cal_frame, "Recalibrating... Look straight", (20, 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.imshow("Concentration Tracker", cal_frame)
            cv2.waitKey(1)
        
        if calib_positions:
            calib_array = np.array(calib_positions)
            baseline_x, baseline_y = np.mean(calib_array, axis=0)
            print(f"Recalibration complete. New baseline: ({baseline_x:.1f}, {baseline_y:.1f})")
        else:
            print("Recalibration failed. Using previous baseline.")

# Cleanup
cap.release()
cv2.destroyAllWindows()

# Session summary
print(f"\n--- Session Summary ---")
print(f"Total Frames: {frame_count}")
print(f"Total Blinks Detected: {blink_counter}")
print(f"Final Concentration: {smooth_score}%")
print("Session ended.")