# Real-Time Eye Tracking Experiment

A comprehensive PsychoPy experiment for real-time eye tracking with webcam-based pupil detection and calibration.

## Overview

This experiment implements a complete eye tracking solution using standard webcams and computer vision techniques. It includes:

- Real-time pupil detection using OpenCV and Haar cascades
- Interactive calibration procedure
- Coordinate transformation from webcam space to screen space
- Gaze recording with visual feedback
- Data logging to CSV format

## Requirements

- Python 3.6+
- OpenCV (`cv2`)
- NumPy
- PsychoPy
- Webcam with clear view of participant's face

## How It Works

1. **Initialization**: The experiment starts by initializing the webcam and loading Haar cascade classifiers for face and eye detection.

2. **Calibration Phase**:
   - Displays targets at 9 predefined screen positions
   - Captures multiple pupil position samples at each target
   - Computes an affine transformation between raw pupil coordinates and screen coordinates

3. **Recording Phase**:
   - Detects pupil position in real-time
   - Maps raw coordinates to screen coordinates using the calibration transform
   - Provides visual feedback with a green dot at the estimated gaze position
   - Logs timestamped gaze data to a CSV file

## Usage

Run the experiment with:

```bash
python main_experiment.py
```

## Notes

- The calibration phase is interactive and requires the participant to fixate on targets at the specified screen positions.
- The recording phase continues until the participant looks away from the screen or the timer runs out.
- The CSV file is saved with timestamped gaze data in the format: `timestamp,x,y`

## Future Improvements

- Integrate more advanced eye detection models (e.g., Convolutional Neural Networks)
- Add support for different screen resolutions and aspect ratios
- Implement head pose estimation for more robust tracking
- Include noise reduction and smoothing techniques
- Add support for different screen resolutions and aspect ratios
