"""Re-run the person detector alone around each target release of a recorded run.

For every release ("target_released_after_grace") it finds where the target
box disappeared, runs the detector on the recorded frames around that moment
and joins the simulator's ground truth (presenter position and box) by time.

Run: python tools/investigate_releases.py NAME [NAME ...] --output DIR
"""
from __future__ import annotations

import argparse
import csv
import datetime
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tracker_run_metrics import FIRST_STATUS, ROOT, _seconds  # noqa: E402


def load_run(name):
    rows = [json.loads(line) for line in (ROOT / "traces" / f"{name}.jsonl").open(encoding="utf-8")]
    start = rows[0]["timestamp"]
    times = np.array([row["timestamp"] - start for row in rows])
    log_start = None
    for line in (ROOT / "logs" / f"{name}.log").open(encoding="utf-8", errors="replace"):
        match = FIRST_STATUS.match(line)
        if match:
            log_start = _seconds(match.group(1), match.group(2))
            break
    truth = []
    with (ROOT / "unity_metrics" / f"{name}_frames.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            epoch = datetime.datetime.fromisoformat(row["utc"].replace("Z", "+00:00")).timestamp()
            truth.append((epoch - log_start, row))
    return rows, times, truth


def truth_at(truth, truth_times, seconds):
    index = min(int(np.searchsorted(truth_times, seconds)), len(truth) - 1)
    return truth[index][1]


def target_box(row):
    target = row["decision"].get("target_track_id")
    for detection in row["detections"]:
        if detection["label"] == "person" and detection["track_id"] == target:
            return detection["box"]
    return None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("names", nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="yolo26n_ncnn_model")
    parser.add_argument("--imgsz", type=int, default=416)
    parser.add_argument("--before", type=float, default=3.0, help="Seconds before the box disappeared")
    parser.add_argument("--after", type=float, default=4.0)
    args = parser.parse_args(argv)
    from ultralytics import YOLO
    model = YOLO(args.model)
    args.output.mkdir(parents=True, exist_ok=True)
    events = []
    for name in args.names:
        rows, times, truth = load_run(name)
        truth_times = np.array([t for t, _ in truth])
        released = [i for i, row in enumerate(rows) if row["decision"]["reason"] == "target_released_after_grace"]
        released = [i for n, i in enumerate(released) if n == 0 or times[i] - times[released[n - 1]] > 1]
        capture = cv2.VideoCapture(str(ROOT / "raw" / f"{name}.mp4"))
        for release in released:
            # Walk back to the last frame that still had the target's box.
            last_seen = release
            while last_seen > 0 and target_box(rows[last_seen]) is None:
                last_seen -= 1
            first = int(np.searchsorted(times, times[last_seen] - args.before))
            last = int(np.searchsorted(times, min(times[release] + 1.0, times[last_seen] + args.after)))
            capture.set(cv2.CAP_PROP_POS_FRAMES, first)
            samples = []
            for index in range(first, min(last, len(rows))):
                ok, frame = capture.read()
                if not ok:
                    break
                result = model.predict(frame, conf=0.01, classes=[0], imgsz=args.imgsz, device="cpu", verbose=False)[0]
                boxes = result.boxes.cpu().numpy()
                state = truth_at(truth, truth_times, times[index])
                best = int(np.argmax(boxes.conf)) if len(boxes.conf) else None
                sample = {"frame": index, "t": float(times[index]), "since_last_seen": float(times[index] - times[last_seen]),
                          "state": rows[index]["decision"]["state"],
                          "tracker_people": len([d for d in rows[index]["detections"] if d["label"] == "person"]),
                          "best_confidence": float(boxes.conf[best]) if best is not None else 0.0,
                          "best_box": [float(v) for v in boxes.xyxy[best]] if best is not None else None,
                          "truth_in_view": state["presenter_in_view"] == "1",
                          "truth_box": [float(state[k]) for k in ("box_x0", "box_y0", "box_x1", "box_y1")],
                          "truth_xz": [float(state["presenter_x"]), float(state["presenter_z"])],
                          "truth_marker_visible": state["marker_visible"] == "1",
                          "pan_tilt": [float(state["pan_deg"]), float(state["tilt_deg"])]}
                samples.append(sample)
                offset = round(sample["since_last_seen"], 1)
                if abs(sample["since_last_seen"]) < 0.04 or any(abs(sample["since_last_seen"] - v) < 0.035 for v in (-1.0, 0.5, 1.5, 2.5)):
                    view = frame.copy()
                    x0, y0, x1, y1 = [int(v) for v in sample["truth_box"]]
                    cv2.rectangle(view, (x0, y0), (x1, y1), (0, 200, 255), 1)
                    if sample["best_box"]:
                        bx0, by0, bx1, by1 = [int(v) for v in sample["best_box"]]
                        cv2.rectangle(view, (bx0, by0), (bx1, by1), (0, 255, 0), 1)
                    cv2.putText(view, f"{offset:+.1f}s conf {sample['best_confidence']:.2f}", (6, 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    cv2.imwrite(str(args.output / f"{name}_release{int(times[release])}_{offset:+.1f}.jpg"), view)
            events.append({"name": name, "release_seconds": float(times[release]),
                           "last_seen_seconds": float(times[last_seen]), "samples": samples})
    (args.output / "releases.json").write_text(json.dumps(events, indent=1))
    print(json.dumps({"releases": len(events)}))


if __name__ == "__main__":
    main()
