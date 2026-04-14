# Real-Time Hand Gesture Recognition System
> Last automated login update: 2026-04-14 12:41:38

Production-ready real-time hand gesture recognition using OpenCV, MediaPipe, and TensorFlow.

## Features

- Real-time hand landmark detection
- Multi-hand tracking and bounding boxes
- Static gesture classification with confidence score
- Prediction smoothing and confidence filtering
- Dynamic swipe gesture detection (up, down, left, right)
- FPS overlay for performance monitoring
- Dataset collection pipeline
- Model training pipeline with metrics plots
- Prediction logging to CSV

## Supported Gestures

- FIST
- PALM
- THUMBS_UP
- PEACE
- OK
- STOP

## Project Structure

```text
hand_gesture_recognition/
  main.py
  collect_data.py
  train_model.py
  hand_detection/
    hand_detector.py
  gesture_recognition/
    gesture_classifier.py
  dataset/
    collect_data.py
  training/
    train_model.py
  models/
  utils/
    draw.py
    fps.py
    logger.py
  config/
    config.yaml
  output/
    predictions_log.csv
  requirements.txt
```

## Installation

```bash
pip install -r requirements.txt
```

## Run Modes

### 1) Data Collection Mode

Collect labeled landmark features into CSV.

```bash
python collect_data.py
```

Controls:
- Press `1`..`6` to switch gesture label
- Press `q` to quit

### 2) Training Mode

Train the deep learning model and save artifacts.

```bash
python train_model.py
```

Outputs:
- `models/gesture_model.h5`
- `models/gesture_labels.json`
- `output/training_metrics.png`

### 3) Inference Mode

Run real-time recognition from webcam.

```bash
python main.py
```

Optional video file input:

```bash
python main.py --video path/to/video.mp4
```

## Configuration

Edit `config/config.yaml`:

```yaml
camera_index: 0
confidence_threshold: 0.6
max_num_hands: 2
model_path: models/gesture_model.h5
labels_path: models/gesture_labels.json
log_path: output/predictions_log.csv
frame_width: 960
frame_height: 540
use_gpu_if_available: true
prediction_smoothing_window: 5
dynamic_window: 8
```

## Notes

- Train a model before running inference.
- Improve accuracy by collecting balanced samples across all classes under varied lighting and backgrounds.
