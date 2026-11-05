#!/usr/bin/env python3
"""Replay a recorded run through the current identity/target logic (open loop).

A run recorded with --record-input-video and --identity-trace-file stores one
video frame and one trace line per processed frame. This tool feeds each
frame, its recorded detections (boxes, track IDs, markers) and its recorded
timestamp back through SpeakerIdentityManager and MarkerTracker, so a logic
change can be checked against a real run without the camera or the detector.

Only the decisions are replayed. The camera does not move, so the PTZ command
counts printed at the end are not a measure of real PTZ behaviour.

    python tools/replay_identity_trace.py room_01            # summary
    python tools/replay_identity_trace.py room_01 -v         # plus identity logs
"""
from __future__ import annotations

import argparse
import collections
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2  # noqa: E402

from marker_tracker import app  # noqa: E402
from marker_tracker.camera import QonHttpCameraAdapter  # noqa: E402
from marker_tracker.core import BoundingBox, Detection, MarkerTracker  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("run_name", help="e.g. room_01 (test_video/raw/<name>.mp4 + traces/<name>.jsonl)")
    parser.add_argument("-v", "--verbose", action="store_true", help="show identity INFO logs")
    parser.add_argument("--max-events", type=int, default=400)
    options, app_argv = parser.parse_known_args()
    logging.basicConfig(
        level=logging.INFO if options.verbose else logging.WARNING,
        format="%(message)s",
    )
    video_path = f"test_video/raw/{options.run_name}.mp4"
    # Same options as the live run; extra app options can be appended on the command line.
    args = app.parse_args(
        [
            "--runtime-profile", "pi-qon-http",
            "--person-tracker", "decoupled",
            "--tracker-config", "botsort-pi.yaml",
            "--speaker-identity-backend", "yolo-reid",
            "--speaker-reid-model", "yolo26n-reid.onnx",
            "--speaker-reid-device", "cpu",
            "--video-file", video_path,
            "--camera-control", "log",
            *app_argv,
        ]
    )
    with open(f"test_video/traces/{options.run_name}.jsonl", encoding="utf-8") as stream:
        records = [json.loads(line) for line in stream]
    capture = cv2.VideoCapture(video_path)
    identity = app._create_speaker_identity_manager(args)
    tracker = MarkerTracker(
        confirmation_seconds=args.confirmation_seconds,
        release_grace_seconds=args.release_grace_seconds,
        marker_dropout_seconds=args.marker_dropout_seconds,
        switch_score_margin=args.switch_score_margin,
        switch_cooldown_seconds=args.switch_cooldown_seconds,
        area_growth_penalty_threshold=args.area_growth_penalty_threshold,
        recovery_confirmation_seconds=args.recovery_confirmation_seconds,
        recovery_min_iou=args.recovery_min_iou,
        recovery_max_center_distance=args.recovery_max_center_distance,
        recovery_score_margin=args.recovery_score_margin,
        recovery_appearance_weight=args.speaker_reid_weight,
        recovery_min_appearance=args.speaker_reid_min_similarity,
        recovery_appearance_margin=args.speaker_reid_margin,
        reacquire_marker_observations=args.reacquire_marker_observations,
        reacquire_marker_window_seconds=args.reacquire_marker_window_seconds,
        reacquire_marker_min_seconds=(
            0.0 if args.disable_bystander_rules else args.reacquire_marker_min_seconds
        ),
        bystander_rules=not args.disable_bystander_rules,
    )
    framing = app.FramingPolicy(args.framing_mode)
    clock = [0.0]
    sent: list[tuple[float, str, int]] = []

    class RecordingCamera(QonHttpCameraAdapter):
        def _send(self, command: str, speed_x: int, speed_y: int) -> None:
            del speed_y
            sent.append((clock[0], command, speed_x))

    camera = RecordingCamera(
        "http://replay",
        min_speed=args.qon_min_speed,
        max_speed=args.qon_max_speed,
        deadzone=args.qon_deadzone,
        stop_deadzone=args.qon_stop_deadzone,
        stop_debounce_seconds=args.qon_stop_debounce_seconds,
        lost_target_hold_seconds=args.qon_lost_target_hold_seconds,
        command_min_interval_seconds=args.qon_command_min_interval_seconds,
        command_ttl_seconds=args.qon_command_ttl_seconds,
        configure_supervisor_actuator=False,
        async_commands=False,
        full_speed_error=args.qon_full_speed_error,
        uniform_error_scale=args.qon_uniform_error_scale,
        axis_switch_ratio=args.qon_axis_switch_ratio,
        response_delay_seconds=args.qon_response_delay_seconds,
        pan_rate_per_speed=args.qon_pan_rate_per_speed,
        tilt_rate_per_speed=args.qon_tilt_rate_per_speed,
        now_fn=lambda: clock[0],
    )

    started_at = records[0]["timestamp"]
    states: collections.Counter[str] = collections.Counter()
    events: list[str] = []
    previous_key = None
    identity_ms: list[float] = []
    for record in records:
        ok, frame = capture.read()
        if not ok:
            break
        now = record["timestamp"]
        clock[0] = now
        detections = [
            Detection(item["label"], item["confidence"], BoundingBox(*item["box"]), item["track_id"])
            for item in record["detections"]
        ]
        identity_started_at = time.perf_counter()
        detections = identity.score_detections(frame, detections, now)
        decision = tracker.update(
            detections,
            now,
            identity_resolution=identity.resolution_context(),
        )
        decision = identity.observe(decision, detections, frame, now)
        identity_ms.append((time.perf_counter() - identity_started_at) * 1000)
        camera.update(
            app._camera_target_for_decision(decision, framing, frame.shape),
            frame.shape,
            speed_limit=app._camera_speed_limit_for_decision(decision, args),
            immediate_stop=decision.state != "grace",
        )
        states[decision.state] += 1
        key = (decision.state, decision.reason, decision.target_track_id)
        if key != previous_key:
            events.append(
                f"{now - started_at:7.2f} {decision.state:18s} {decision.reason:36s} "
                f"target={decision.target_track_id} handoff={decision.emit_handoff} "
                f"release={decision.emit_release}"
            )
            previous_key = key

    print("\n".join(events[: options.max_events]))
    print("events", len(events), "frames per state", dict(states))
    print("ptz commands (open loop)", len(sent), dict(collections.Counter(item[1] for item in sent)))
    identity_ms.sort()
    print(
        "identity ms avg %.1f p95 %.1f max %.1f"
        % (
            sum(identity_ms) / len(identity_ms),
            identity_ms[int(len(identity_ms) * 0.95)],
            identity_ms[-1],
        )
    )


if __name__ == "__main__":
    main()
