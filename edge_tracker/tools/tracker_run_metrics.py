"""Metrics for one tracker run from its identity trace, log and raw recording.

Run: python tools/tracker_run_metrics.py NAME [NAME ...]
Reads test_video/traces/NAME.jsonl, test_video/logs/NAME.log and
test_video/raw/NAME.mp4 and prints one JSON object per run.
"""
from __future__ import annotations

import collections
import datetime
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent / "test_video"
PTZ_SEND = re.compile(r"(\S+ \S+),(\d+) INFO Qon HTTP PTZ send GET .*ptzcmd&(\w+)&(\d+)")
FIRST_STATUS = re.compile(r"(\S+ \S+),(\d+) INFO Tracking status: fps=0.0")
OPPOSITE = ({"left", "right"}, {"up", "down"})


def is_reversal(first, second):
    """Two consecutive move commands that push one axis in opposite directions.

    Works for single-axis commands (left -> right) and diagonal ones
    (leftup -> rightup reverses the pan).
    """
    for negative, positive in (("left", "right"), ("up", "down")):
        a = -1 if negative in first else 1 if positive in first else 0
        b = -1 if negative in second else 1 if positive in second else 0
        if a * b < 0:
            return True
    return False


def _seconds(date_time, milliseconds):
    return datetime.datetime.strptime(date_time, "%Y-%m-%d %H:%M:%S").timestamp() + int(milliseconds) / 1000


def _background_shift(previous, current, rows_pair):
    """Shift of the image centre caused by camera motion, people masked out."""
    mask = np.full(current.shape, 255, np.uint8)
    for row in rows_pair:
        for detection in row["detections"]:
            x1, y1, x2, y2 = [int(v) for v in detection["box"]]
            mask[max(0, y1 - 10):y2 + 10, max(0, x1 - 15):x2 + 15] = 0
    points = cv2.goodFeaturesToTrack(previous, 400, 0.01, 7, mask=mask)
    if points is None or len(points) < 15:
        return None
    moved, ok, _ = cv2.calcOpticalFlowPyrLK(previous, current, points, None, winSize=(25, 25), maxLevel=3)
    back, ok_back, _ = cv2.calcOpticalFlowPyrLK(current, previous, moved, None, winSize=(25, 25), maxLevel=3)
    keep = (ok.ravel() > 0) & (ok_back.ravel() > 0) & (np.linalg.norm((points - back).reshape(-1, 2), axis=1) < 1)
    if keep.sum() < 15:
        return None
    model, _ = cv2.estimateAffinePartial2D(points[keep], moved[keep], method=cv2.RANSAC, ransacReprojThreshold=1.5)
    if model is None:
        return None
    height, width = current.shape
    centre = model @ np.array([width / 2, height / 2, 1.0])
    return centre[0] - width / 2, centre[1] - height / 2


def response_times(times, shift, commands, minimum_seconds=0.8):
    """How long the picture takes to reach full speed after a pan command.

    Uses pan commands sent while the camera was stopped and then left
    unchanged for at least `minimum_seconds`. The steady speed is the median
    over the later part of that stretch; reported are the times from the
    command to 10% (delay) and to 90% of it, and their difference (response).
    The frame rate limits the resolution to about one frame interval.
    """
    with np.errstate(invalid="ignore", divide="ignore"):
        per_frame = np.abs(shift[1:] / np.diff(times))
    frame_times = times[1:]
    runs = []
    for at, command, speed in commands:
        if runs and runs[-1][1] == command and (command == "ptzstop" or runs[-1][2] == speed):
            continue  # a refresh of the command already in effect
        runs.append((at, command, speed))
    events = []
    for index in range(1, len(runs) - 1):
        at, command, speed = runs[index]
        end = runs[index + 1][0]
        if command not in ("left", "right") or runs[index - 1][1] != "ptzstop":
            continue
        if at - runs[index - 1][0] < 0.3 or end - at < minimum_seconds:
            continue
        later = (frame_times >= at + 0.6) & (frame_times <= end) & ~np.isnan(per_frame)
        if later.sum() < 3:
            continue
        steady = float(np.median(per_frame[later]))
        if steady < 15:
            continue
        window = (frame_times >= at) & (frame_times <= end) & ~np.isnan(per_frame)
        reached = lambda share: next((float(t - at) for t, v in zip(frame_times[window], per_frame[window]) if v >= share * steady), None)
        t10, t90 = reached(0.1), reached(0.9)
        if t10 is not None and t90 is not None:
            events.append({"level": speed, "steady_px_s": steady, "to_10_percent": t10, "to_90_percent": t90})
    if not events:
        return {"events": 0}
    column = lambda key: [event[key] for event in events]
    rise = [event["to_90_percent"] - event["to_10_percent"] for event in events]
    return {"events": len(events),
            "delay_seconds_median": float(np.median(column("to_10_percent"))),
            "to_90_percent_seconds_median": float(np.median(column("to_90_percent"))),
            "response_seconds_median": float(np.median(rise)),
            "response_seconds_quartiles": [float(np.percentile(rise, 25)), float(np.percentile(rise, 75))]}


def metrics(name, with_video=True):
    rows = [json.loads(line) for line in (ROOT / "traces" / f"{name}.jsonl").open(encoding="utf-8")]
    start = rows[0]["timestamp"]
    times = np.array([row["timestamp"] - start for row in rows])
    duration = float(times[-1])
    log_start, commands = None, []
    for line in (ROOT / "logs" / f"{name}.log").open(encoding="utf-8", errors="replace"):
        match = PTZ_SEND.match(line)
        if match:
            commands.append((_seconds(match.group(1), match.group(2)), match.group(3), int(match.group(4))))
        match = FIRST_STATUS.match(line)
        if match and log_start is None:
            log_start = _seconds(match.group(1), match.group(2))
    commands = [(t - log_start, command, speed) for t, command, speed in commands]
    states = collections.Counter(row["decision"]["state"] for row in rows)
    result = {"name": name, "duration_seconds": duration, "frames": len(rows), "states": dict(states),
              "marker_frames": sum(any(d["label"] == "marker" for d in row["detections"]) for row in rows)}
    registered = [row["timestamp"] - start for row in rows if row["decision"]["reason"] == "confirmed_marker_handoff"]
    result["registered_at_seconds"] = registered[0] if registered else None
    released = [row["timestamp"] - start for row in rows if row["decision"]["reason"] == "target_released_after_grace"]
    result["lost_at_seconds"] = [t for i, t in enumerate(released) if i == 0 or t - released[i - 1] > 1]
    errors = []
    for row in rows:
        decision = row["decision"]
        if decision["state"] != "tracking" or decision.get("target_track_id") is None:
            continue
        for detection in row["detections"]:
            if detection["label"] == "person" and detection["track_id"] == decision["target_track_id"]:
                x1, y1, x2, y2 = detection["box"]
                # Vertical error is measured at the upper-body framing point (FramingPolicy).
                errors.append((((x1 + x2) / 2 - 320) / 640, (y1 + 0.35 * (y2 - y1) - 180) / 360))
    if len(errors) < 50:
        result["valid"] = False
        return result
    errors = np.abs(np.array(errors))
    moves = [command for _, command, _ in commands if command != "ptzstop"]
    reversals = sum(1 for a, b in zip(moves, moves[1:]) if is_reversal(a, b))
    level = np.zeros(len(rows))
    index = -1
    for i, t in enumerate(times):
        while index + 1 < len(commands) and commands[index + 1][0] <= t:
            index += 1
        if index >= 0:
            _, command, speed = commands[index]
            level[i] = speed if "left" in command else -speed if "right" in command else 0
    pan_time = collections.Counter(int(abs(v)) for v in level if v)
    total = sum(pan_time.values()) or 1
    result.update(
        valid=True, tracking_share=states["tracking"] / len(rows),
        x_within_10=float((errors[:, 0] <= 0.10).mean()), x_within_15=float((errors[:, 0] <= 0.15).mean()),
        x_median=float(np.median(errors[:, 0])), x_p90=float(np.percentile(errors[:, 0], 90)),
        x_max=float(errors[:, 0].max()),
        y_framing_within_10=float((errors[:, 1] <= 0.10).mean()), y_framing_p90=float(np.percentile(errors[:, 1], 90)),
        reversals_per_minute=reversals / duration * 60, commands_per_minute=len(commands) / duration * 60,
        pan_level_time_share={str(k): v / total for k, v in sorted(pan_time.items())})
    if not with_video:
        return result
    capture = cv2.VideoCapture(str(ROOT / "raw" / f"{name}.mp4"))
    shift = np.full(len(rows), np.nan)
    shift_y = np.full(len(rows), np.nan)
    previous = None
    for i in range(min(int(capture.get(cv2.CAP_PROP_FRAME_COUNT)), len(rows))):
        ok, frame = capture.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if previous is not None:
            value = _background_shift(previous, gray, (rows[i - 1], rows[i]))
            if value is not None:
                shift[i], shift_y[i] = value
        previous = gray
    position = np.nancumsum(shift)
    speed = np.full(len(rows), np.nan)
    for i in range(2, len(rows) - 1):
        if not np.isnan(shift[i - 1:i + 2]).any():
            speed[i] = (position[i + 1] - position[i - 2]) / (times[i + 1] - times[i - 2])
    valid = ~np.isnan(speed)
    # Delay between a command and the picture moving: best lag of command level against image speed.
    grid = np.arange(0, duration, 0.01)
    level_grid, speed_grid = np.interp(grid, times, level), np.interp(grid, times[valid], speed[valid])
    correlation = [np.corrcoef(level_grid[:len(level_grid) - k], speed_grid[k:])[0, 1] for k in range(120)]
    lag = int(np.argmax(correlation)) * 0.01
    result["command_to_motion_seconds"] = lag
    # Image speed per command level (pixels/second at this frame size): the
    # table the tracker's --qon-level-speeds options take. Only stretches where
    # one command stayed in effect for more than 0.25 s count, shifted by the lag.
    position_y = np.nancumsum(shift_y)
    speed_y = np.full(len(rows), np.nan)
    for i in range(2, len(rows) - 1):
        if not np.isnan(shift_y[i - 1:i + 2]).any():
            speed_y[i] = (position_y[i + 1] - position_y[i - 2]) / (times[i + 1] - times[i - 2])
    names = ["ptzstop"] * len(rows)
    index = -1
    for i, t in enumerate(times):
        while index + 1 < len(commands) and commands[index + 1][0] <= t:
            index += 1
        if index >= 0 and commands[index][1] in ("left", "right", "up", "down"):
            names[i] = f"{commands[index][1]}:{commands[index][2]}"
    tables = {"pan": collections.defaultdict(list), "tilt": collections.defaultdict(list)}
    begin = 0
    for i in range(1, len(rows) + 1):
        if i == len(rows) or names[i] != names[begin]:
            if names[begin] != "ptzstop" and times[i - 1] - times[begin] > 0.25:
                direction, step = names[begin].split(":")
                axis = "pan" if direction in ("left", "right") else "tilt"
                chosen = (times >= times[begin] + 0.25 + lag) & (times <= times[i - 1] + lag)
                values = np.abs((speed if axis == "pan" else speed_y)[chosen])
                tables[axis][int(step)].extend(values[~np.isnan(values)].tolist())
            begin = i
    result["response"] = response_times(times, shift, commands)
    result["level_speeds_px_s"] = {
        axis: {str(step): {"median": float(np.median(values)), "q1": float(np.percentile(values, 25)),
                           "q3": float(np.percentile(values, 75)), "samples": len(values)}
               for step, values in sorted(table.items()) if len(values) >= 3}
        for axis, table in tables.items()}
    return result


if __name__ == "__main__":
    for run_name in sys.argv[1:]:
        print(json.dumps(metrics(run_name)))
