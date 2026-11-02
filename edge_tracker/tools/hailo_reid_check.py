#!/usr/bin/env python3
"""Check speaker ReID on the Hailo NPU against a recorded run (run this on the Pi).

Runs the detector HEF and the ReID HEF on one shared device, the way the app
does with --hailo-hef and --speaker-reid-hef, over frames of a recorded run.
Prints the time per call and how well the NPU embeddings separate the
recorded target from the other people, for comparison with the float model
on the development PC (room_01: target about 0.87, others about 0.59).

    python tools/hailo_reid_check.py room_01
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from marker_tracker.core import BoundingBox, Detection  # noqa: E402
from marker_tracker.detection import HailoAppearanceEncoder, HailoPersonDetector  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("run_name")
    parser.add_argument("--hailo-hef", default="yolo26n_h8.hef")
    parser.add_argument("--speaker-reid-hef", default="repvgg_a0_person_reid_512_h8.hef")
    parser.add_argument("--every", type=int, default=4, help="use every Nth frame")
    options = parser.parse_args()

    detector = HailoPersonDetector(options.hailo_hef, shared_device=True)
    encoder = HailoAppearanceEncoder(options.speaker_reid_hef)
    with open(f"test_video/traces/{options.run_name}.jsonl", encoding="utf-8") as stream:
        records = [json.loads(line) for line in stream]
    capture = cv2.VideoCapture(f"test_video/raw/{options.run_name}.mp4")

    detect_ms, encode_ms, samples = [], [], []
    for index, record in enumerate(records):
        ok, frame = capture.read()
        if not ok:
            break
        if index % options.every:
            continue
        started_at = time.perf_counter()
        detector.detect(frame, 0.1)
        detect_ms.append((time.perf_counter() - started_at) * 1000)
        decision = record["decision"]
        people = [
            item
            for item in record["detections"]
            if item["label"] == "person"
            and item["track_id"] is not None
            and item["box"][3] - item["box"][1] >= 60
        ]
        if decision["state"] != "tracking" or not people:
            continue
        detections = [Detection("person", 1.0, BoundingBox(*item["box"])) for item in people]
        started_at = time.perf_counter()
        features = encoder.encode(frame, detections)
        encode_ms.append((time.perf_counter() - started_at) * 1000 / len(detections))
        for item, feature in zip(people, features):
            samples.append(
                (item["track_id"] == decision["target_track_id"], feature / np.linalg.norm(feature))
            )

    targets = [feature for is_target, feature in samples if is_target]
    others = [feature for is_target, feature in samples if not is_target]
    gallery = np.stack(targets[:12])
    score = lambda feature: float(np.sort(gallery @ feature)[-3:].mean())  # noqa: E731
    target_scores = [score(feature) for feature in targets[12:]]
    other_scores = [score(feature) for feature in others]
    print("frames", len(detect_ms), "target crops", len(targets), "other crops", len(others))
    print("detect ms  median %.1f p95 %.1f" % (statistics.median(detect_ms), np.percentile(detect_ms, 95)))
    print("reid ms per crop  median %.1f p95 %.1f" % (statistics.median(encode_ms), np.percentile(encode_ms, 95)))
    print("embedding size", len(targets[0]))
    print("target similarity  median %.3f p10 %.3f" % (statistics.median(target_scores), np.percentile(target_scores, 10)))
    if other_scores:
        print("others similarity  median %.3f p90 %.3f" % (statistics.median(other_scores), np.percentile(other_scores, 90)))


if __name__ == "__main__":
    main()
