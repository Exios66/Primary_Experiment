#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Enhanced PsychoPy Experiment File for Real-Time Eye Tracking with Calibration

This experiment:
  - Uses OpenCV to capture webcam frames.
  - Detects the subject's face and eyes using Haar cascades.
  - Processes the eye region to localize the pupil with advanced techniques.
  - Runs a robust calibration phase with configurable parameters.
  - Computes an affine transformation mapping raw pupil coordinates to screen coordinates.
  - Provides real-time gaze feedback with configurable visualization options.
  - Records timestamped gaze data to a CSV file with additional metrics.

Note:
This implementation includes enhanced algorithms for more accurate pupil detection
and improved calibration to handle head movements and varying lighting conditions.
"""

import cv2
import numpy as np
import csv
import time
import os
from psychopy import visual, core, event, gui, monitors
from scipy.spatial import distance
import matplotlib.pyplot as plt
from datetime import datetime
import sys
import keyboard
from psychopy.hardware import keyboard as psych_keyboard
import warnings

# Initialize Haar cascade classifiers (using OpenCV's built-in cascades)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")

# Configuration parameters
CONFIG = {
    'pupil_detection': {
        'blur_kernel': (7, 7),
        'threshold_min': 20,  # Lower for darker pupils
        'threshold_max': 255,
        'min_pupil_size': 10,  # Minimum area of pupil contour
        'max_pupil_size': 2000  # Maximum area of pupil contour
    },
    'calibration': {
        'target_size': 0.05,
        'target_color': 'red',
        'samples_per_target': 7,
        'fixation_time': 1.0,
        'sample_delay': 0.1
    },
    'recording': {
        'sample_rate': 30,  # Hz
        'feedback_enabled': True,
        'feedback_size': 0.03,
        'feedback_color': 'green'
    },
    'debug': {
        'enabled': False,  # Set to True to enable debug mode
        'save_eye_images': False
    }
}

# Add system-specific configuration
SYSTEM_CONFIG = {
    'keyboard': {
        'allowed_keys': ['c', 'r', 'q', 'escape', 'return', 'backspace', 'k'] + list('0123456789'),
        'quit_keys': ['q', 'escape'],
        'confirm_keys': ['return', 'space'],
        'backspace_keys': ['backspace'],
        'killswitch_key': 'k'  # New killswitch key
    },
    'display': {
        'resolution': (1920, 1080),
        'refresh_rate': 60,
        'background_color': (0.2, 0.2, 0.2),
        'fullscreen': True
    },
    'timing': {
        'frame_tolerance': 0.001,
        'keyboard_timeout': 0.1
    },
    'killswitch': {
        'enabled': True,
        'button_size': 0.05,
        'button_color': 'red',
        'button_pos': (0.95, 0.95)  # Top-right corner
    }
}

def ensure_output_dir():
    """Create output directory for data and debug images if it doesn't exist."""
    output_dir = os.path.join(os.getcwd(), 'eye_tracking_data')
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if CONFIG['debug']['enabled'] and CONFIG['debug']['save_eye_images']:
        debug_dir = os.path.join(output_dir, 'debug_images')
        if not os.path.exists(debug_dir):
            os.makedirs(debug_dir)
    return output_dir

def init_camera(camera_id=0, width=640, height=480, fps=30):
    """
    Initialize the camera with specific parameters.
    
    Args:
        camera_id: Camera device ID (default: 0 for primary camera)
        width: Frame width
        height: Frame height
        fps: Frames per second
        
    Returns:
        OpenCV VideoCapture object
    """
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        raise Exception(f"Could not open camera ID {camera_id}.")
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    
    # Verify settings
    actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"Camera initialized with resolution {actual_width}x{actual_height} @ {actual_fps} FPS")
    return cap

def detect_pupil(frame, debug=False):
    """
    Enhanced pupil detection with improved robustness and error handling.
    
    Args:
        frame: Input video frame
        debug: Whether to enable debug visualizations
        
    Returns:
        (pupil_x, pupil_y, pupil_size) in pixel coordinates, or None if detection fails
    """
    # Initialize debug images if needed
    debug_images = {}
    if debug:
        debug_images['original'] = frame.copy()
    
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if debug:
        debug_images['grayscale'] = gray.copy()
    
    # Equalize histogram to improve contrast
    equalized = cv2.equalizeHist(gray)
    if debug:
        debug_images['equalized'] = equalized.copy()
    
    # Detect face
    faces = face_cascade.detectMultiScale(
        equalized, 
        scaleFactor=1.1, 
        minNeighbors=5, 
        minSize=(30, 30)
    )
    
    if len(faces) == 0:
        return None
    
    # Sort faces by area (largest first) and use the largest face
    faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)
    (x, y, w, h) = faces[0]
    
    # Draw face rectangle if debug mode
    if debug:
        debug_images['face_detected'] = frame.copy()
        cv2.rectangle(debug_images['face_detected'], (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    # Extract face ROI
    face_roi = equalized[y:y+h, x:x+w]
    
    # Detect eyes in the face region
    eyes = eye_cascade.detectMultiScale(
        face_roi,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )
    
    if len(eyes) == 0:
        return None
    
    # Sort eyes by y-coordinate and use the upper one (usually more reliable)
    eyes = sorted(eyes, key=lambda e: e[1])
    
    for eye_idx, (ex, ey, ew, eh) in enumerate(eyes[:2]):  # Try up to 2 eye regions
        # Extract eye ROI
        eye_roi = face_roi[ey:ey+eh, ex:ex+ew]
        
        if debug:
            debug_images[f'eye_{eye_idx}'] = cv2.cvtColor(eye_roi, cv2.COLOR_GRAY2BGR)
            
        # Apply Gaussian blur to reduce noise
        cfg = CONFIG['pupil_detection']
        eye_blur = cv2.GaussianBlur(eye_roi, cfg['blur_kernel'], 0)
        
        # Apply adaptive thresholding to identify dark regions (pupil)
        _, thresh = cv2.threshold(
            eye_blur, 
            cfg['threshold_min'], 
            cfg['threshold_max'], 
            cv2.THRESH_BINARY_INV
        )
        
        if debug:
            debug_images[f'eye_{eye_idx}_thresh'] = thresh.copy()
        
        # Find contours in the thresholded image
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            continue
        
        # Filter contours by size and shape
        valid_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if cfg['min_pupil_size'] <= area <= cfg['max_pupil_size']:
                # Calculate circularity to ensure it's roughly circular
                perimeter = cv2.arcLength(contour, True)
                circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
                
                # Pupil should be relatively circular (circularity close to 1)
                if circularity > 0.5:
                    valid_contours.append((contour, area, circularity))
        
        if not valid_contours:
            continue
            
        # Sort by area (largest) and circularity
        valid_contours.sort(key=lambda x: (x[1], x[2]), reverse=True)
        best_contour = valid_contours[0][0]
        
        # Calculate pupil center using moments
        M = cv2.moments(best_contour)
        if M['m00'] == 0:
            continue
            
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        
        # Calculate pupil size (radius)
        pupil_size = np.sqrt(cv2.contourArea(best_contour) / np.pi)
        
        if debug:
            debug_images[f'eye_{eye_idx}_result'] = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
            cv2.circle(debug_images[f'eye_{eye_idx}_result'], (cx, cy), int(pupil_size), (0, 255, 0), 2)
        
        # Convert eye ROI coordinates to frame coordinates
        pupil_x = x + ex + cx
        pupil_y = y + ey + cy
        
        # Save debug images if configured
        if CONFIG['debug']['enabled'] and CONFIG['debug']['save_eye_images']:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            output_dir = os.path.join(ensure_output_dir(), 'debug_images')
            cv2.imwrite(os.path.join(output_dir, f'eye_{timestamp}.png'), debug_images.get(f'eye_{eye_idx}', thresh))
            cv2.imwrite(os.path.join(output_dir, f'eye_thresh_{timestamp}.png'), debug_images.get(f'eye_{eye_idx}_thresh', thresh))
            if f'eye_{eye_idx}_result' in debug_images:
                cv2.imwrite(os.path.join(output_dir, f'eye_result_{timestamp}.png'), debug_images[f'eye_{eye_idx}_result'])
        
        return (pupil_x, pupil_y, pupil_size)
    
    return None

def show_message(win, text, duration=2, wait_for_key=False):
    """Enhanced message display with keyboard interaction option."""
    message = visual.TextStim(
        win,
        text=text,
        height=0.07,
        color="white",
        wrapWidth=1.8,
        alignText='center'
    )
    
    message.draw()
    win.flip()
    
    if wait_for_key:
        event.waitKeys()
    else:
        core.wait(duration)

def draw_calibration_target(win, position, size=None, color=None):
    """Draw a calibration target with an optional inner point for better fixation."""
    # Use configuration parameters if not specified
    size = size or CONFIG['calibration']['target_size']
    color = color or CONFIG['calibration']['target_color']
    
    # Draw outer circle
    outer_circle = visual.Circle(
        win, 
        radius=size,
        pos=position,
        fillColor=None,
        lineColor=color,
        lineWidth=2
    )
    
    # Draw inner point for precise fixation
    inner_point = visual.Circle(
        win,
        radius=size/5,
        pos=position,
        fillColor=color,
        lineWidth=0
    )
    
    outer_circle.draw()
    inner_point.draw()
    
    # Return components for animation if needed
    return outer_circle, inner_point

def calibration_procedure(win, cap, n_samples=None):
    """
    Enhanced calibration procedure with better visual feedback and data validation.
    
    Args:
        win: PsychoPy window
        cap: OpenCV VideoCapture object
        n_samples: Number of samples per target (uses CONFIG if None)
        
    Returns:
        transform: A calibration transform function
        accuracy: Mean calibration accuracy (euclidean distance)
    """
    # Use config if n_samples not specified
    n_samples = n_samples or CONFIG['calibration']['samples_per_target']
    fixation_time = CONFIG['calibration']['fixation_time']
    sample_delay = CONFIG['calibration']['sample_delay']
    
    # Define calibration target positions (normalized coordinates)
    calib_targets = [
        (-0.8, 0.8), (0, 0.8), (0.8, 0.8),
        (-0.8, 0),   (0, 0),   (0.8, 0),
        (-0.8, -0.8),(0, -0.8),(0.8, -0.8)
    ]
    
    raw_points = []
    screen_points = []
    calibration_data = []  # Store detailed calibration data
    
    # Show instructions
    instructions = visual.TextStim(
        win, 
        text="Eye Tracking Calibration\n\nPlease look at each target as it appears.\n"\
             "Try to keep your head still during calibration.\n\n"\
             "Press any key to begin.",
        height=0.07, 
        color="white",
        wrapWidth=1.8
    )
    instructions.draw()
    win.flip()
    event.waitKeys()  # Wait for keypress to start
    
    # Progress bar for visual feedback
    progress_bar = visual.Rect(
        win, 
        width=1.6, 
        height=0.05, 
        pos=(0, -0.9),
        fillColor=None,
        lineColor="white"
    )
    progress = visual.Rect(
        win, 
        width=0, 
        height=0.05, 
        pos=(-0.8, -0.9),
        fillColor="white",
        lineColor=None,
        anchorHoriz='left'
    )
    progress_text = visual.TextStim(
        win,
        text="0%",
        height=0.05,
        pos=(0, -0.9),
        color="white"
    )
    
    # Calibration loop
    for i, target in enumerate(calib_targets):
        # Update progress bar
        progress_percent = i / len(calib_targets)
        progress.width = 1.6 * progress_percent
        progress_text.text = f"{int(progress_percent * 100)}%"
        
        # Animation for target appearance (grows from small to target size)
        for size_factor in np.linspace(0.2, 1.0, 10):
            size = CONFIG['calibration']['target_size'] * size_factor
            outer, inner = draw_calibration_target(win, target, size=size)
            progress_bar.draw()
            progress.draw()
            progress_text.draw()
            win.flip()
            core.wait(0.02)
        
        # Draw final target
        outer, inner = draw_calibration_target(win, target)
        progress_bar.draw()
        progress.draw()
        progress_text.draw()
        win.flip()
        
        # Wait for user to fixate on the target
        core.wait(fixation_time)
        
        # Collect samples
        samples = []
        sample_count = 0
        max_attempts = n_samples * 3  # Allow more attempts to get good samples
        attempts = 0
        
        while sample_count < n_samples and attempts < max_attempts:
            ret, frame = cap.read()
            if not ret:
                attempts += 1
                continue
                
            # Toggle debug mode for more detailed pupil detection
            pupil_data = detect_pupil(frame, debug=CONFIG['debug']['enabled'])
            
            if pupil_data is not None:
                # We now receive (x, y, size) from detect_pupil
                samples.append(pupil_data)
                sample_count += 1
                
                # Visual feedback: briefly flash the target to indicate sample captured
                if sample_count < n_samples:
                    outer, inner = draw_calibration_target(win, target, color="green")
                    progress_bar.draw()
                    progress.draw()
                    progress_text.draw()
                    win.flip()
                    core.wait(0.05)
                    
                    outer, inner = draw_calibration_target(win, target)
                    progress_bar.draw()
                    progress.draw()
                    progress_text.draw()
                    win.flip()
            
            attempts += 1
            core.wait(sample_delay)
        
        # Calculate average pupil position for this target
        if len(samples) > 0:
            # Extract just coordinates (x, y) from samples for averaging
            coords = np.array([(s[0], s[1]) for s in samples])
            sizes = np.array([s[2] for s in samples])
            
            # Calculate mean and standard deviation
            avg_raw = np.mean(coords, axis=0)
            std_raw = np.std(coords, axis=0)
            avg_size = np.mean(sizes)
            
            # Store for calibration
            raw_points.append(avg_raw)
            screen_points.append(np.array(target))
            
            # Store detailed data for verification
            calibration_data.append({
                'target': target,
                'raw_mean': avg_raw,
                'raw_std': std_raw,
                'pupil_size': avg_size,
                'samples': len(samples),
                'attempts': attempts
            })
        else:
            print(f"Warning: Could not collect samples for target {target}")
    
    # Final progress update
    progress.width = 1.6
    progress_text.text = "100%"
    progress_bar.draw()
    progress.draw()
    progress_text.draw()
    win.flip()
    
    # Error handling: ensure we have enough points for calibration
    if len(raw_points) < 6:
        show_message(win, "Calibration failed: Too few valid points.\nPlease try again.", duration=3)
        return None, 0.0
    
    # Convert to numpy arrays
    raw_points = np.array(raw_points)  # shape: (n,2)
    screen_points = np.array(screen_points)  # shape: (n,2)
    
    # Compute affine transformation
    # We augment raw_points with ones to solve: screen = A * raw + b
    N = raw_points.shape[0]
    X_design = np.hstack([raw_points, np.ones((N, 1))])  # shape: (N,3)
    
    # Solve separately for x and y screen coordinates
    coef_x, residuals_x, _, _ = np.linalg.lstsq(X_design, screen_points[:,0], rcond=None)
    coef_y, residuals_y, _, _ = np.linalg.lstsq(X_design, screen_points[:,1], rcond=None)
    
    # Define the transformation function
    def transform(raw_coords):
        """Transform raw pupil coordinates to screen coordinates."""
        # Handle both single coordinates and arrays of coordinates
        is_single = not hasattr(raw_coords[0], '__iter__')
        
        if is_single:
            raw = np.array(raw_coords[:2])  # Extract just x,y (ignore size if present)
            x_aug = np.append(raw, 1)
            screen_x = np.dot(coef_x, x_aug)
            screen_y = np.dot(coef_y, x_aug)
            return (screen_x, screen_y)
        else:
            # Handle multiple coordinates at once
            raw = np.array([r[:2] for r in raw_coords])  # Extract just x,y from each
            x_aug = np.column_stack([raw, np.ones(raw.shape[0])])
            screen_x = np.dot(x_aug, coef_x)
            screen_y = np.dot(x_aug, coef_y)
            return np.column_stack([screen_x, screen_y])
    
    # Evaluate calibration accuracy
    predicted = transform(raw_points)
    errors = np.sqrt(np.sum((predicted - screen_points)**2, axis=1))
    mean_error = np.mean(errors)
    
    # Display calibration results
    accuracy_message = f"Calibration complete.\nAverage error: {mean_error:.3f} units."
    show_message(win, accuracy_message, duration=2)
    
    # Save calibration data
    output_dir = ensure_output_dir()
    calibration_file = os.path.join(output_dir, f"calibration_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    
    with open(calibration_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['target_x', 'target_y', 'raw_x', 'raw_y', 'std_x', 'std_y', 'pupil_size', 'error'])
        
        for i, data in enumerate(calibration_data):
            target = data['target']
            raw = data['raw_mean']
            std = data['raw_std']
            pred = transform(raw)
            error = distance.euclidean(pred, target)
            
            writer.writerow([
                target[0], target[1],
                raw[0], raw[1],
                std[0], std[1],
                data['pupil_size'],
                error
            ])
    
    # Return the transformation function and mean error
    return transform, mean_error

def create_killswitch_button(win):
    """Create a killswitch button in the top-right corner."""
    if not SYSTEM_CONFIG['killswitch']['enabled']:
        return None
        
    button = visual.Circle(
        win,
        radius=SYSTEM_CONFIG['killswitch']['button_size'],
        pos=SYSTEM_CONFIG['killswitch']['button_pos'],
        fillColor=SYSTEM_CONFIG['killswitch']['button_color'],
        lineColor='white',
        lineWidth=2,
        opacity=0.7
    )
    
    label = visual.TextStim(
        win,
        text='X',
        pos=SYSTEM_CONFIG['killswitch']['button_pos'],
        color='white',
        height=SYSTEM_CONFIG['killswitch']['button_size'],
        bold=True
    )
    
    return (button, label)

def check_killswitch(win, kb, mouse, killswitch_elements=None):
    """
    Check if killswitch has been activated (either by keyboard 'K' or clicking the button).
    
    Args:
        win: PsychoPy window
        kb: Keyboard object
        mouse: Mouse object
        killswitch_elements: Tuple of (button, label) for the killswitch UI
        
    Returns:
        bool: True if killswitch activated, False otherwise
    """
    # Check keyboard (K key)
    keys = kb.getKeys(keyList=[SYSTEM_CONFIG['keyboard']['killswitch_key']])
    if keys:
        return True
    
    # Check mouse click on killswitch button
    if killswitch_elements and mouse.getPressed()[0]:  # Left click
        button, _ = killswitch_elements
        if button.contains(mouse.getPos()):
            return True
    
    return False

def safe_exit(components, message="Experiment terminated by killswitch"):
    """
    Safely terminate the experiment and return to PsychoPy interface.
    
    Args:
        components: Dictionary of experiment components
        message: Optional message to display before exiting
    """
    try:
        if components and 'window' in components and components['window'] is not None:
            # Display exit message
            exit_msg = visual.TextStim(
                components['window'],
                text=message + "\n\nPress any key to exit...",
                height=0.07,
                color="white",
                wrapWidth=1.8,
                alignText='center'
            )
            exit_msg.draw()
            components['window'].flip()
            core.wait(0.5)  # Brief pause to prevent accidental key press
            event.waitKeys()  # Wait for any key
        
        # Cleanup
        cleanup_experiment(components)
        
    except Exception as e:
        print(f"Error during safe exit: {str(e)}")
    finally:
        sys.exit(0)

def record_gaze(win, cap, transform, csv_filename=None, duration=10, killswitch_fn=None):
    """Enhanced gaze recording with killswitch support."""
    # Generate output filename if not provided
    if csv_filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = ensure_output_dir()
        csv_filename = os.path.join(output_dir, f"gaze_data_{timestamp}.csv")
    
    # Create visual elements for recording feedback
    feedback_dot = visual.Circle(
        win, 
        radius=CONFIG['recording']['feedback_size'], 
        fillColor=CONFIG['recording']['feedback_color']
    )
    
    recording_text = visual.TextStim(
        win,
        text=f"Recording gaze: 0/{duration}s",
        pos=(0, -0.9),
        height=0.05,
        color="white"
    )
    
    timer_bar = visual.Rect(
        win,
        width=1.6,
        height=0.05,
        pos=(0, -0.8),
        fillColor=None,
        lineColor="white"
    )
    
    timer_progress = visual.Rect(
        win,
        width=0,
        height=0.05,
        pos=(-0.8, -0.8),
        fillColor="white",
        lineColor=None,
        anchorHoriz='left'
    )
    
    # Setup gaze path visualization (shows recent gaze trail)
    max_trail_points = 20
    gaze_trail = []
    trail_visual = visual.ElementArrayStim(
        win,
        elementTex=None,
        elementMask="circle",
        nElements=max_trail_points,
        sizes=0.01,
        colors=[0.5, 1.0, 0.5],  # Light green
        opacities=0.0,  # Start fully transparent
        xys=[(0,0)] * max_trail_points
    )
    
    with open(csv_filename, mode='w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        # Enhanced CSV header
        csv_writer.writerow([
            "timestamp", 
            "gaze_x", 
            "gaze_y", 
            "pupil_size", 
            "confidence", 
            "raw_x", 
            "raw_y"
        ])
        
        start_time = time.time()
        frame_count = 0
        detection_success_count = 0
        sample_interval = 1.0 / CONFIG['recording']['sample_rate']
        
        # Use a consistent timestamp baseline
        baseline_time = start_time
        
        while time.time() - start_time < duration:
            # Check killswitch if provided
            if killswitch_fn and killswitch_fn():
                return {'killed': True}
            
            current_time = time.time()
            elapsed = current_time - start_time
            
            # Update timer visuals
            progress_fraction = elapsed / duration
            timer_progress.width = 1.6 * progress_fraction
            recording_text.text = f"Recording gaze: {int(elapsed)}/{duration}s"
            
            # Control frame rate
            if frame_count > 0:
                time_since_last = current_time - last_frame_time
                if time_since_last < sample_interval:
                    core.wait(sample_interval - time_since_last)
            
            last_frame_time = time.time()
            frame_count += 1
            
            # Capture and process frame
            ret, frame = cap.read()
            if not ret:
                continue
            
            # Detect pupil
            pupil_data = detect_pupil(frame, debug=CONFIG['debug']['enabled'])
            
            # Draw base visualization elements
            timer_bar.draw()
            timer_progress.draw()
            recording_text.draw()
            
            if pupil_data is not None:
                detection_success_count += 1
                raw_x, raw_y, pupil_size = pupil_data
                
                # Apply calibration
                gaze_x, gaze_y = transform((raw_x, raw_y))
                
                # Calculate simple detection confidence
                confidence = min(1.0, detection_success_count / max(1, frame_count))
                
                # Clamp coordinates to visible range (-1 to 1)
                gaze_x = max(-1.0, min(1.0, gaze_x))
                gaze_y = max(-1.0, min(1.0, gaze_y))
                
                # Update data and write to CSV
                timestamp = current_time - baseline_time
                csv_writer.writerow([
                    timestamp, 
                    gaze_x, 
                    gaze_y, 
                    pupil_size, 
                    confidence,
                    raw_x,
                    raw_y
                ])
                
                # Update gaze visualization
                if CONFIG['recording']['feedback_enabled']:
                    # Update gaze trail
                    gaze_trail.append((gaze_x, gaze_y))
                    if len(gaze_trail) > max_trail_points:
                        gaze_trail = gaze_trail[-max_trail_points:]
                    
                    # Set opacity based on recency (more recent = more opaque)
                    opacities = np.linspace(0.2, 1.0, len(gaze_trail))
                    
                    # Update trail visualization
                    trail_visual.xys = gaze_trail + [(0,0)] * (max_trail_points - len(gaze_trail))
                    trail_visual.opacities = np.concatenate([opacities, np.zeros(max_trail_points - len(gaze_trail))])
                    trail_visual.draw()
                    
                    # Draw current gaze point
                    feedback_dot.pos = (gaze_x, gaze_y)
                    feedback_dot.draw()
            
            # Update display
            win.flip()
    
    # Calculate and display recording stats
    success_rate = detection_success_count / frame_count if frame_count > 0 else 0
    sample_rate = frame_count / duration
    
    stats_msg = (
        f"Recording complete.\n"
        f"Success rate: {success_rate:.1%}\n"
        f"Average sample rate: {sample_rate:.1f} Hz\n"
        f"Data saved to: {os.path.basename(csv_filename)}"
    )
    
    show_message(win, stats_msg, duration=3)
    
    return {
        'filename': csv_filename,
        'duration': duration,
        'frames': frame_count,
        'detections': detection_success_count,
        'success_rate': success_rate,
        'sample_rate': sample_rate
    }

def visualize_calibration_results(transform, raw_points, screen_points):
    """
    Create and display a visualization of calibration accuracy.
    
    Args:
        transform: Calibration transformation function
        raw_points: Raw pupil coordinates
        screen_points: Target screen coordinates
    """
    # Convert raw points to screen coordinates using the transform
    predicted = transform(raw_points)
    
    # Create a figure
    plt.figure(figsize=(10, 8))
    
    # Plot target points (blue)
    plt.scatter(screen_points[:, 0], screen_points[:, 1], c='blue', s=100, label='Targets')
    
    # Plot predicted points (red)
    plt.scatter(predicted[:, 0], predicted[:, 1], c='red', s=50, label='Predicted')
    
    # Draw lines connecting corresponding points
    for i in range(len(screen_points)):
        plt.plot([screen_points[i, 0], predicted[i, 0]], 
                 [screen_points[i, 1], predicted[i, 1]], 
                 'k-', alpha=0.3)
    
    # Add labels and title
    plt.xlabel('X Coordinate (Normalized)')
    plt.ylabel('Y Coordinate (Normalized)')
    plt.title('Calibration Results: Target vs. Predicted Gaze Positions')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Set axis limits to match normalized coordinates
    plt.xlim(-1.1, 1.1)
    plt.ylim(-1.1, 1.1)
    
    # Calculate and display average error
    errors = np.sqrt(np.sum((predicted - screen_points)**2, axis=1))
    mean_error = np.mean(errors)
    plt.annotate(f'Mean Error: {mean_error:.4f}', 
                 xy=(0.05, 0.05), 
                 xycoords='figure fraction')
    
    # Save figure
    output_dir = ensure_output_dir()
    plt.savefig(os.path.join(output_dir, f"calibration_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"))
    
    # Display figure (non-blocking)
    plt.show(block=False)
    plt.pause(0.1)

def init_experiment():
    """
    Initialize all experiment components with proper error handling.
    
    Returns:
        dict: Contains initialized components (window, keyboard, etc.)
    """
    components = {}
    
    try:
        # Create experiment info dialog
        exp_info = {
            'participant': '',
            'session': '001',
            'debug_mode': False
        }
        
        dlg = gui.DlgFromDict(
            dictionary=exp_info,
            title='Eye Tracking Experiment',
            fixed=['debug_mode']
        )
        
        if not dlg.OK:
            print("User cancelled experiment.")
            core.quit()
        
        # Update debug configuration
        CONFIG['debug']['enabled'] = exp_info['debug_mode']
        
        # Initialize monitor
        mon = monitors.Monitor('default')
        mon.setSizePix(SYSTEM_CONFIG['display']['resolution'])
        mon.setWidth(30)  # Set physical width in cm
        mon.saveMon()
        
        # Create window with proper error handling
        win = visual.Window(
            fullscr=SYSTEM_CONFIG['display']['fullscreen'],
            monitor=mon,
            color=SYSTEM_CONFIG['display']['background_color'],
            units="norm",
            allowGUI=False,
            winType='pyglet',
            checkTiming=True
        )
        
        # Check if window creation was successful
        if not win._isFullScr and SYSTEM_CONFIG['display']['fullscreen']:
            warnings.warn("Could not create fullscreen window. Falling back to windowed mode.")
        
        # Initialize keyboard
        kb = psych_keyboard.Keyboard()
        
        # Test keyboard connectivity
        test_keys = event.getKeys()
        if not test_keys:
            print("Warning: No keyboard input detected. Please check keyboard connectivity.")
        
        # Initialize camera with error handling
        cap = init_camera()
        if not cap.isOpened():
            raise RuntimeError("Failed to initialize camera. Please check camera connectivity.")
        
        # Initialize mouse for killswitch button
        mouse = event.Mouse(win=win)
        
        # Create killswitch button
        killswitch_elements = create_killswitch_button(win)
        
        # Store components
        components.update({
            'window': win,
            'keyboard': kb,
            'camera': cap,
            'exp_info': exp_info,
            'monitor': mon,
            'mouse': mouse
        })
        
        return components
        
    except Exception as e:
        cleanup_experiment(components)
        raise RuntimeError(f"Failed to initialize experiment: {str(e)}")

def cleanup_experiment(components):
    """
    Safely cleanup all experiment components.
    
    Args:
        components: Dictionary containing experiment components
    """
    try:
        if 'camera' in components and components['camera'] is not None:
            components['camera'].release()
        
        if 'window' in components and components['window'] is not None:
            components['window'].close()
        
        core.quit()
        
    except Exception as e:
        print(f"Error during cleanup: {str(e)}")
        sys.exit(1)

def handle_keyboard_input(kb, allowed_keys=None, timeout=None):
    """
    Handle keyboard input with proper error checking.
    
    Args:
        kb: PsychoPy keyboard object
        allowed_keys: List of allowed keys
        timeout: Maximum time to wait for input
        
    Returns:
        list: List of detected keys
    """
    if allowed_keys is None:
        allowed_keys = SYSTEM_CONFIG['keyboard']['allowed_keys']
    
    if timeout is None:
        timeout = SYSTEM_CONFIG['timing']['keyboard_timeout']
    
    # Clear keyboard buffer
    kb.clearEvents()
    
    # Wait for valid input
    keys = []
    start_time = core.getTime()
    
    while not keys and (timeout is None or core.getTime() - start_time < timeout):
        keys = kb.getKeys(keyList=allowed_keys, waitRelease=False)
        
        if 'escape' in [key.name for key in keys]:
            return ['escape']
            
    return [key.name for key in keys]

def main():
    """Enhanced main experiment function with killswitch functionality."""
    components = None
    
    try:
        # Initialize all components
        components = init_experiment()
        win = components['window']
        kb = components['keyboard']
        cap = components['camera']
        
        # Initialize mouse for killswitch button
        mouse = components['mouse']
        
        # Create killswitch button
        killswitch_elements = create_killswitch_button(win)
        
        # Display welcome message
        welcome_text = """
        Eye Tracking Experiment
        
        This experiment will track your eye movements using a webcam.
        
        Press:
        C - Calibrate the eye tracker
        R - Record gaze data (requires calibration first)
        Q - Quit the experiment
        K - Emergency stop (killswitch)
        
        Press any key to begin...
        """
        
        welcome = visual.TextStim(
            win,
            text=welcome_text,
            height=0.07,
            color="white",
            wrapWidth=1.8,
            alignText='center'
        )
        
        # Main experiment loop
        transform = None
        running = True
        
        while running:
            # Show welcome screen and killswitch
            welcome.draw()
            if killswitch_elements:
                killswitch_elements[0].draw()
                killswitch_elements[1].draw()
            win.flip()
            
            # Check for killswitch activation
            if check_killswitch(win, kb, mouse, killswitch_elements):
                safe_exit(components)
            
            # Wait for valid input
            keys = handle_keyboard_input(kb)
            
            if not keys:
                continue
            
            if 'escape' in keys or 'q' in keys:
                running = False
                continue
            
            elif 'c' in keys:
                # Run calibration
                show_message(win, "Starting calibration procedure...\nPress any key when ready.", wait_for_key=True)
                transform, accuracy = calibration_procedure(win, cap)
                
                if transform is None:
                    show_message(win, "Calibration failed. Please try again.", duration=2)
                else:
                    show_message(win, f"Calibration successful!\nAccuracy: {accuracy:.3f}", duration=2)
                
            elif 'r' in keys:
                if transform is None:
                    show_message(win, "Please calibrate first (press 'C').", duration=2)
                    continue
                
                # Get recording duration
                duration = get_recording_duration(win, kb)
                if duration is None:
                    continue
                
                # Start recording with killswitch monitoring
                show_message(
                    win,
                    f"Recording eye movements for {duration} seconds...\n"
                    "Please look around the screen naturally.\n\n"
                    "Press any key to begin... (K to cancel)",
                    wait_for_key=True
                )
                
                # Start recording with killswitch monitoring
                try:
                    record_results = record_gaze(win, cap, transform, duration=duration, 
                                               killswitch_fn=lambda: check_killswitch(win, kb, mouse, killswitch_elements))
                    
                    if record_results.get('killed', False):
                        show_message(win, "Recording cancelled by killswitch.", duration=2)
                        continue
                        
                    show_message(
                        win,
                        f"Recording completed.\n"
                        f"Success rate: {record_results['success_rate']:.1%}\n"
                        f"Sample rate: {record_results['sample_rate']:.1f} Hz\n\n"
                        "Press any key to continue...",
                        wait_for_key=True
                    )
                except Exception as e:
                    show_message(win, f"Error during recording: {str(e)}\nPress any key to continue...", wait_for_key=True)
        
        # Clean exit
        cleanup_experiment(components)
        
    except Exception as e:
        if components:
            safe_exit(components, message=f"An error occurred:\n{str(e)}")
        else:
            print(f"Error in experiment: {str(e)}")
            sys.exit(1)

def get_recording_duration(win, kb):
    """Get recording duration with proper input validation."""
    duration_prompt = visual.TextStim(
        win,
        text="Enter recording duration (10-60 seconds):\nPress Enter to confirm, Escape to cancel",
        height=0.07,
        color="white",
        pos=(0, 0.2)
    )
    
    duration_input = visual.TextStim(
        win,
        text="30",
        height=0.07,
        color="yellow",
        pos=(0, -0.2)
    )
    
    duration_str = "30"
    done = False
    
    while not done:
        duration_prompt.draw()
        duration_input.draw()
        win.flip()
        
        keys = handle_keyboard_input(kb)
        
        if 'escape' in keys:
            return None
        elif 'return' in keys:
            done = True
        elif 'backspace' in keys:
            duration_str = duration_str[:-1] if duration_str else ""
        else:
            for key in keys:
                if key in '0123456789' and len(duration_str) < 2:
                    duration_str += key
        
        duration_input.text = duration_str
    
    try:
        duration = int(duration_str)
        return max(10, min(60, duration))
    except ValueError:
        return 30

if __name__ == "__main__":
    main()