# PTZ Marker Tracking MVP

This is a graduation-project MVP for marker-selected PTZ auto-tracking. It reads
frames from a webcam, RTSP stream, or video file; detects people and ArUco
markers; selects the person whose bbox contains the marker center; and sends
coarse PTZ simulator commands from the selected person's bbox center.

## Run

```bash
pip install -r requirements.txt
python3 main_marker_tracking_demo.py --source 0
python3 main_marker_tracking_demo.py --source path/to/video.mp4
python3 main_marker_tracking_demo.py --source rtsp://user:pass@camera/stream
```

Use `--no-window` for headless simulator output, `--gesture` to enable optional
gesture fallback, and `--target-marker-id 7` to acquire only one known marker id.
Other marker ids are still drawn in the debug view, but they are ignored for
target registration.

For deterministic smoke tests or controlled demo footage, use
`--detector manual --person-box x,y,w,h` to provide a known person bbox.

## ArUco Marker

Generate a printable marker:

```bash
python3 tools/generate_aruco_marker.py --id 7 --size 800 --output marker_7.png
```

The generator uses `DICT_4X4_50` by default. Print the marker with a clear white
border, keep it flat, and attach it to the target person's chest where it stays
visible to the camera. For a stable presentation demo, run with the matching id:

```bash
python3 main_marker_tracking_demo.py --source 0 --target-marker-id 7
```

## Detector Backends

Choose a detector explicitly with `--detector {hog,opencv-yolo,ncnn,manual}`.
`hog` is only a fallback when no project detector is available.

OpenCV DNN/ONNX warning: `YoloOpenCVPersonDetector` is a temporary demo adapter.
YOLO ONNX output layouts differ by exporter and model version, so its parser must
be checked against the exact model before relying on it.

Raspberry Pi 5 NCNN example:

```bash
python3 main_marker_tracking_demo.py \
  --source 0 \
  --detector ncnn \
  --ncnn-param models/yolo.param \
  --ncnn-bin models/yolo.bin \
  --ncnn-input-size 640 \
  --conf-threshold 0.35 \
  --nms-threshold 0.45 \
  --target-marker-id 7
```

If the existing project already has NCNN YOLO decode/NMS logic, reuse it by
passing that parser into `NCNNPersonDetector(output_parser=...)`. The built-in
generic NCNN parser is intentionally conservative and expects rows like
`[x1, y1, x2, y2, score, class_id]`; model-specific heads should be decoded by
the project adapter.

## Existing Detector Adapter

Keep the existing YOLO/ByteTrack detector unchanged and wrap its output:

```python
from person_detector import DetectionResultAdapter, ExistingProjectPersonDetectorAdapter

person_detector = ExistingProjectPersonDetectorAdapter(existing_detector, bbox_format="xyxy")
people = person_detector.detect(frame)
```

The adapter accepts ByteTrack-style `tlwh`, YOLO-style `xyxy`, dict/object
outputs, and simple rows like `[x1, y1, x2, y2, score, class_id]`. It returns the
demo's `PersonDetection(bbox=(x, y, w, h), confidence=score)` objects.

## Keyboard Controls

- `q`: quit
- `r`: reset target
- `s`: stop tracking
- `m`: toggle marker-only mode
- `g`: enable/disable gesture fallback

## MVP Behavior

- The active target is acquired only when an ArUco marker center is inside a
  person bbox.
- PTZ movement uses the selected person bbox center, not the marker center.
- A dead zone around frame center prevents small jitter commands.
- When the marker briefly disappears, the fallback bbox tracker keeps the same
  target if the person remains near the previous bbox. If no match is found, the
  state becomes `SUSPENDED` before resetting after `--max-suspended-frames`.
- Gesture fallback is only used when no marker target is active. If MediaPipe is
  installed and exactly one detected person has a raised wrist above shoulder
  level for one second, that person is registered as a temporary target.

Real hardware integration belongs in `HardwarePTZController` in
`ptz_controller.py`; the current implementation intentionally prints simulator
commands for a reliable demo without a PTZ camera.
