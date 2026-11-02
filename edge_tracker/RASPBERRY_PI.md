# Raspberry Pi 5 NCNN Deployment

This project is prepared to run the detector with the exported NCNN model:

```text
yolo26n_ncnn_model/
```

## Install

On Raspberry Pi 5:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-pi.txt
```

## Export Again

The exported model is already present. To regenerate it:

```bash
pip install -r requirements-export.txt
python tools/export_ncnn_model.py --model yolo26n.pt --imgsz 416
```

This creates or updates:

```text
yolo26n_ncnn_model/
```

## Run With Qon HTTP Profile

For the current Raspberry Pi 5 + Qon HTTP test path, use the deployment profile
instead of passing every tuning option by hand:

```bash
python -m marker_tracker.app \
  --runtime-profile pi-qon-http \
  --rtsp-url "rtsp://USER:PASSWORD@CAMERA_IP:554/STREAM"
```

The profile derives `--qon-camera-url` from the RTSP host when both URLs use the
same camera address. Add `--qon-camera-url` only if the PTZ HTTP endpoint uses a
different host. Add `--no-display` for headless stability tests, and add
`--qon-log-commands` only while debugging HTTP requests and responses.

First retest the closest path to the old working ByteTrack build. This still
uses Ultralytics `model.track()`/ByteTrack, but uses the original `.pt` model
path and does not force `imgsz` unless explicitly requested:

```bash
python -m marker_tracker.app \
  --runtime-profile pi-qon-http \
  --model yolo26n.pt \
  --rtsp-url "rtsp://USER:PASSWORD@CAMERA_IP:554/STREAM" \
  --no-display
```

Then retest the default NCNN profile with the matched NCNN image size:

```bash
python -m marker_tracker.app \
  --runtime-profile pi-qon-http \
  --rtsp-url "rtsp://USER:PASSWORD@CAMERA_IP:554/STREAM" \
  --no-display
```

If the process still aborts in the Ultralytics ByteTrack/Kalman path, bypass
`model.track()` and use `model.predict()` plus the in-process IoU tracker:

```bash
python -m marker_tracker.app \
  --runtime-profile pi-qon-http \
  --rtsp-url "rtsp://USER:PASSWORD@CAMERA_IP:554/STREAM" \
  --person-tracker simple-iou \
  --no-display
```

The profile applies:

```text
model=yolo26n_ncnn_model
imgsz=416
bytetrack_config=bytetrack-pi.yaml
aruco_marker_id=0
aruco_detect_flip=unmirrored
confirmation_seconds=2.0
marker_dropout_seconds=0.75
camera_control=qon-http
framing_mode=upper-body
```

Add only the values you want to override, for example:

```bash
python -m marker_tracker.app \
  --runtime-profile pi-qon-http \
  --rtsp-url "rtsp://USER:PASSWORD@CAMERA_IP:554/STREAM" \
  --confirmation-seconds 1.5
```

## Run With VISCA Reference Path

VISCA support is kept as a fallback/reference path for comparing older camera
tests. The current Qon field path should use the HTTP profile above.

Replace `CAMERA_IP` with the PTZ camera address.

```bash
python -m marker_tracker.app \
  --model yolo26n_ncnn_model \
  --aruco-dictionary DICT_4X4_50 \
  --aruco-marker-id 0 \
  --aruco-detect-flip auto \
  --confirmation-seconds 1.0 \
  --camera-control visca \
  --visca-host CAMERA_IP \
  --visca-port 52381 \
  --visca-log-commands
```

## Input Size

For NCNN models, keep `--imgsz` matched to the export size. The included `yolo26n_ncnn_model` is currently exported at 416, so
the HTTP profile defaults to 416:

```bash
python -m marker_tracker.app \
  --runtime-profile pi-qon-http \
  --rtsp-url "rtsp://USER:PASSWORD@CAMERA_IP:554/STREAM" \
  --imgsz 416
```

To use a smaller inference size such as 576 or 512, re-export the NCNN model at
that size first and then run with the matching `--imgsz`.

The options below apply only when using the VISCA reference path.

If the VISCA camera does not move, retry with raw VISCA UDP payloads:

```bash
--visca-packet-mode raw
```

If movement direction is reversed:

```bash
--visca-invert-pan
--visca-invert-tilt
```

If tracking jitters, increase the deadzone:

```bash
--visca-deadzone-x 0.18 --visca-deadzone-y 0.18
```

If tracking reacts too slowly, increase:

```bash
--visca-pan-speed 12 --visca-tilt-speed 10
```

## NPU person detection (Hailo)

`--person-tracker decoupled` splits the person source into a detector stage and
a tracker stage (`marker_tracker/detection.py`). The tracker stage drives the
same Ultralytics BYTETracker/BoT-SORT classes with the same tracker YAML as
`model.track()`, but takes its boxes from any `PersonDetector`:

```python
class PersonDetector(Protocol):
    def detect(self, frame, confidence: float) -> list[Detection]: ...
```

Without `--hailo-hef` the detector is `UltralyticsPersonDetector`
(`model.predict()` on the CPU). With `--hailo-hef PATH` it is
`HailoPersonDetector`, which runs the HEF on the Hailo NPU through HailoRT.
Everything after detection (tracker, ArUco, identity, speaker ReID, PTZ) is the
same code on the CPU.

Install on the Pi (Raspberry Pi OS, AI HAT+ / AI Kit):

```bash
sudo apt install hailo-all        # HailoRT, driver, hailo_platform, sample HEFs
sudo reboot
hailortcli fw-control identify    # must print the Hailo device
python3 -m venv --system-site-packages .venv   # so the venv sees hailo_platform
. .venv/bin/activate
pip install -r requirements-pi.txt
```

Run (add the usual camera options or `--runtime-profile pi-qon-http`):

```bash
python -m marker_tracker.app \
  --runtime-profile pi-qon-http \
  --person-tracker decoupled \
  --tracker-config botsort-pi.yaml \
  --hailo-hef yolo26n_h8.hef \
  --performance-diagnostics \
  --rtsp-url "rtsp://USER:PASSWORD@CAMERA_IP:554/STREAM"
```

`yolo26n_h8.hef` is the Hailo Model Zoo v2.19.0 build of YOLO26n for Hailo-8
(AI HAT+ 26 TOPS), 640x640 input, the same COCO weights as `yolo26n.pt`:
https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.19.0/hailo8/yolo26n.hef
It has no built-in NMS; `HailoPersonDetector` decodes its six raw one-to-one
head outputs. HEFs with built-in NMS post-processing also work, for example
`/usr/share/hailo-models/yolov8s_h8.hef` from `hailo-all`, which is a useful
fallback if the installed HailoRT is too old to load `yolo26n_h8.hef`
(`hailortcli parse-hef yolo26n_h8.hef` shows whether it loads). A HEF only runs
on the chip it was compiled for (`h8` is Hailo-8, not Hailo-8L). With
`--performance-diagnostics` the log adds `person_detect_*_ms` and
`person_track_*_ms`, so CPU and NPU detector latency can be compared directly.

Verified on the development PC:

- With the CPU detector, `decoupled` gives the same boxes and track IDs as
  `ultralytics` on all 414 frames of `crossing_04.mp4` (`botsort-pi.yaml`, no
  ReID).
- The YOLO26 raw-output decoding gives the same person boxes and scores as
  Ultralytics `predict()` when fed the same head outputs from `yolo26n.pt`
  (20 frames of `crossing_04.mp4`, 30 boxes).

Verified on a Raspberry Pi 5 with Hailo-8: the detector HEF ran in the
`room_01`, `room_02` and `pi_unity_03`-`05` runs (`person_detect_avg_ms` about
12-15).

Notes:

- The HEF runs at 640x640 while the NCNN path used 416, and its weights are
  quantized, so confidences can differ slightly from earlier recordings.
- `--model` is still loaded but unused, and the tracker stage still imports
  `ultralytics` (and therefore torch) on the Pi.
- The tracker YAML must have `with_reid: False`; speaker ReID stays in
  `--speaker-identity-backend` and still runs on the CPU.

## NPU speaker ReID (Hailo)

On the Pi the CPU ReID call was the cause of the frame-rate drops in room_01
(120-185 ms per call, 6-12 fps while resolving an identity). `--speaker-reid-hef`
runs ReID on the NPU instead:

```bash
  --speaker-identity-backend yolo-reid \
  --speaker-reid-hef repvgg_a0_person_reid_512_h8.hef \
  --speaker-reid-similarity-scale 1.31 \
  --speaker-reid-similarity-offset -0.537 \
```

`repvgg_a0_person_reid_512_h8.hef` is the Hailo Model Zoo v2.19.0 build for
Hailo-8 (256x128 input, 512 values out):
https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v2.19.0/hailo8/repvgg_a0_person_reid_512.hef

It is a different model from `yolo26n-reid.onnx`, so its cosine similarities
sit on another scale (room_01: target about 0.87 and other people about 0.59,
against 0.60 and 0.24). The two similarity options map its scores linearly
onto the scale the identity thresholds were tuned on; they are required with
`--speaker-reid-hef`, because without them bystanders pass the thresholds.

With both HEFs the detector and the ReID network share one device through the
HailoRT round-robin scheduler. Without `--speaker-reid-hef` the detector keeps
the exclusive device as before.

Verified on the development PC with the float ONNX model on the CPU
(`room_01`, 6764 frames, replayed through the identity and target logic):
the target was the same in every frame in which both models were tracking
(5736), with the same 8 target changes and fewer frames spent resolving
(457 against 547).

The NPU ReID path also ran on the Pi in the `room_02` and `pi_unity_03`-`05`
runs. To check
a new device or a new HEF:

```bash
python tools/hailo_reid_check.py room_01
python tools/replay_identity_trace.py room_01 \
  --speaker-reid-hef repvgg_a0_person_reid_512_h8.hef \
  --speaker-reid-similarity-scale 1.31 --speaker-reid-similarity-offset -0.537 | tail -4
```

The first prints the time per call and the target/others similarity; the
target should be near 0.87 and the others near 0.59.

- The calibration comes from one recording with three people. Treat it as a
  starting point, not as a general constant.

## ArUco flip modes

`aruco_detect_flip=unmirrored` (the profile default) tries the frame and its
180-degree rotation. `auto` also tries the two mirrored versions on every
frame without a marker; a mirrored marker cannot decode, so on an unmirrored
stream those two attempts never succeed (room_01: 0 of 223 sightings) and
double the cost. Use `--aruco-detect-flip auto` only for a mirrored stream.

## Parallel marker detection

ArUco detection reads only the frame, so it runs on a worker thread while the
person detector and tracker handle the same frame. The detections are the
same as before (414 of 414 frames identical on `crossing_04.mp4`); only the
wall time per frame changes. `--serial-target-signal` restores the old order.

In `--performance-diagnostics`, `target_signal_*_ms` is now the marker
thread's own elapsed time, which overlaps `person_source_*_ms` and can read
higher than before because the two compete for the CPU. Compare
`pipeline_*_ms` with and without `--serial-target-signal` to see the gain. On
the development PC (CPU detector) the frame time went from 30.0 to 26.2 ms;
it has not been measured on the Pi.

## Display overlay

The display window defaults to `--overlay simple`: thin boxes without labels
for people, a thick green box for the speaker, a yellow box for the marker,
the camera aim cross, the FPS and one state word (WAITING, REGISTERING,
TRACKING, SEARCHING). TRACKING is held for 0.4 s so one-frame detector
dropouts do not make the word flicker. `--overlay debug` shows the earlier
labels, track IDs, reason and PTZ status. Logs, traces and the recorded input
video are the same in both modes.

## Current Limits

- NCNN is used for person detection only.
- `--person-tracker simple-iou` is a stability fallback for Pi crashes in the
  Ultralytics tracker path. It assigns IDs by IoU, so long occlusions and heavy
  crossings will be less robust than ByteTrack.
- Qon HTTP PTZ support currently sends one-axis continuous pan/tilt commands and
  stop commands through `ptzctrl.cgi`.
- VISCA support is retained as a reference path and currently covers continuous
  pan/tilt drive and stop commands.
