"""Compare tracker-log metrics with the Unity simulator's ground-truth CSV for a test series.

Run: python tools/compare_unity_series.py PREFIX LABEL [LABEL ...] [--repeats 3]
Reads test_video/unity_metrics/PREFIX_LABEL_N_{frames.csv,summary.json,meta.json}
and the tracker's own trace/log/recording, and prints JSON.

Unity values are recomputed from frames.csv over the time the tracker was
actually steering (from the first PTZ move command to the end), weighted by
frame time, so the seconds before the marker is registered do not count.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from investigate_releases import load_run, target_box  # noqa: E402
from tracker_run_metrics import ROOT, is_reversal, metrics  # noqa: E402

MOVES = {"left", "right", "up", "down", "leftup", "leftdown", "rightup", "rightdown"}
# Provisional acceptance limits supplied for this series.
# The reversal limit applies to hunting only (camera flips while the presenter does not).
LIMITS = {"lost": 0, "keep15_min": 0.80, "x_max_below": 0.35, "hunting_per_minute_max": 7.0}
PRESENTER_MOVING_DEG_S = {"pan": 1.0, "tilt": 0.5}


def classify_reversals(rows, moves, total_seconds):
    """Split camera direction changes into following the presenter and hunting.

    The presenter's direction as seen from the camera is the PTZ angle plus the
    angular offset of the target point in the picture, so no camera position is
    needed. A reversal is "follow" when the presenter is now moving the way the
    camera turned and was not doing so shortly before; otherwise it is
    "hunting": the presenter kept its direction, or stood still, and only the
    camera flipped.
    """
    time = np.array([float(row["time"]) for row in rows])
    half = np.tan(np.radians(np.array([float(row["hfov_deg"]) for row in rows]) / 2))
    def column(name):
        return np.array([float(row[name]) if row[name] else np.nan for row in rows])
    pan = column("pan_deg")
    pan = np.where(pan > 180, pan - 360, pan)
    angle = {"pan": pan + np.degrees(np.arctan(2 * column("err_x") * half)),
             "tilt": column("tilt_deg") + np.degrees(np.arctan(2 * column("err_y") * half * 360 / 640))}

    def rate(axis, start, end):
        chosen = (time >= start) & (time <= end) & ~np.isnan(angle[axis])
        if chosen.sum() < 10:
            return None
        return float(np.polyfit(time[chosen], angle[axis][chosen], 1)[0])

    counts = {"follow": 0, "hunting": 0, "unknown": 0}
    events = []
    for axis, (negative, positive) in (("pan", ("left", "right")), ("tilt", ("up", "down"))):
        previous = 0
        for at, direction, _ in moves:
            sign = -1 if negative in direction else 1 if positive in direction else 0
            if sign == 0:
                continue  # a command for the other axis only does not end this axis's run
            if previous and sign != previous:
                now, before = rate(axis, at - 0.7, at + 0.1), rate(axis, at - 2.2, at - 0.7)
                threshold = PRESENTER_MOVING_DEG_S[axis]
                if now is None:
                    kind = "unknown"
                elif now * sign >= threshold and (before is None or before * sign < threshold):
                    kind = "follow"
                else:
                    kind = "hunting"
                counts[kind] += 1
                events.append((at, axis, kind))
            previous = sign
    result = {key: value / total_seconds * 60 for key, value in counts.items()}
    result["events"] = events
    return result


def split_by_presenter_state(rows, start_seconds, reversal_events):
    """The same measures for the time the presenter stands still and the time it walks."""
    chosen = [row for row in rows if float(row["time"]) >= start_seconds and "dwelling" in row]
    if not chosen:
        return None
    time = np.array([float(row["time"]) for row in chosen])
    dt = np.array([float(row["dt"]) for row in chosen])
    dwelling = np.array([row["dwelling"] == "1" for row in chosen])
    visible = np.array([row["presenter_in_view"] == "1" for row in chosen])
    error = np.abs(np.array([float(row["err_x"]) if row["err_x"] else np.nan for row in chosen]))
    hfov = np.array([float(row["hfov_deg"]) for row in chosen])
    upper = np.array([0.5 * (float(row["box_y1"]) - float(row["box_y0"])) / 360 for row in chosen])
    def weighted_median(values, weights):
        keep = ~np.isnan(values)
        order = np.argsort(values[keep])
        cumulative = np.cumsum(weights[keep][order])
        return float(values[keep][order][np.searchsorted(cumulative, cumulative[-1] / 2)]) if keep.any() else None
    out = {}
    for label, part in (("dwelling", dwelling), ("walking", ~dwelling)):
        seconds = float(dt[part].sum())
        if seconds < 1:
            out[label] = {"seconds": seconds}
            continue
        seen = part & visible
        at_state = lambda t: bool(part[min(len(part) - 1, int(np.searchsorted(time, t)))])
        hunting = sum(1 for t, _, kind in reversal_events if kind == "hunting" and at_state(t))
        out[label] = {
            "seconds": seconds,
            "keep15": float(dt[seen & (error <= 0.15)].sum() / seconds),
            "x_median": weighted_median(error[part], dt[part]),
            "x_p90": float(np.nanpercentile(error[part], 90)),
            "hfov_median": weighted_median(hfov[part], dt[part]),
            "upper_ratio_median": weighted_median(upper[seen], dt[seen]) if seen.any() else None,
            "upper_in_range": float(dt[seen & (upper >= 0.4675) & (upper <= 0.6325)].sum() / seconds),
            "out_of_view_seconds": float(dt[part & ~visible].sum()),
            "hunting_per_minute": hunting / seconds * 60,
        }
    return out


def registrations_and_releases(name):
    """Marker size when the target was (re)registered, and what each release looked like."""
    rows, times, truth = load_run(name)
    truth_times = np.array([t for t, _ in truth])
    def truth_at(seconds):
        return truth[min(len(truth) - 1, int(np.searchsorted(truth_times, seconds)))][1]
    registrations, releases = [], []
    released_at = None
    for index, row in enumerate(rows):
        decision = row["decision"]
        if decision.get("emit_handoff") and decision["reason"] in ("confirmed_marker_handoff", "confirmed_marker_transfer"):
            state = truth_at(times[index] - 0.4)
            seen = [d["box"][2] - d["box"][0] for d in row["detections"] if d["label"] == "marker"]
            registrations.append({"at_seconds": float(times[index]), "marker_px_unity": float(state["marker_px"] or 0),
                                  "marker_px_tracker": float(seen[0]) if seen else None,
                                  "hfov": float(state["hfov_deg"]),
                                  "seconds_since_release": None if released_at is None else float(times[index] - released_at)})
            released_at = None
        if decision["reason"] == "target_released_after_grace" and (not releases or times[index] - releases[-1]["at_seconds"] > 1):
            last_seen = index
            while last_seen > 0 and target_box(rows[last_seen]) is None:
                last_seen -= 1
            state = truth_at(times[last_seen] + 0.5 - 0.4)
            in_view = state["presenter_in_view"] == "1"
            releases.append({"at_seconds": float(times[index]), "lost_at_seconds": float(times[last_seen]),
                             "cause": "detection lost while in view" if in_view else "left the picture",
                             "hfov": float(state["hfov_deg"]), "marker_px_unity": float(state["marker_px"] or 0),
                             "marker_visible": state["marker_visible"] == "1",
                             "dwelling": state.get("dwelling") == "1",
                             "truth_box_height_px": float(state["box_y1"]) - float(state["box_y0"])})
            released_at = times[index]
    for release in releases:
        after = [r["at_seconds"] for r in registrations if r["at_seconds"] > release["at_seconds"]]
        resumed = next((float(times[i]) for i in range(len(rows)) if times[i] > release["at_seconds"] and rows[i]["decision"]["state"] == "tracking"), None)
        release["seconds_until_tracking_again"] = None if resumed is None else resumed - release["at_seconds"]
    return {"registrations": registrations, "releases": releases}


# Where the presenter walks behind the wall edge near the camera (Unity x, z in metres).
PILLAR_SPOT = (-3.3, -1.1)
PILLAR_RADIUS_M = 1.0


def grace_episodes(name, min_seconds=0.5):
    """Each stretch in which the target was undetected but not yet released.

    Tells where the presenter was, whether the grace period was extended past
    `release_grace_seconds` (prediction judged inside the picture), how far the
    camera panned meanwhile and how the stretch ended.
    """
    rows, times, truth = load_run(name)
    truth_times = np.array([t for t, _ in truth])
    episodes, start = [], None
    for index in range(len(rows) + 1):
        state = rows[index]["decision"]["state"] if index < len(rows) else None
        if state == "grace":
            start = index if start is None else start
            continue
        if start is None:
            continue
        first, last = start, index - 1
        start = None
        duration = float(times[last] - times[first])
        if duration < min_seconds:
            continue
        # The same target tracked again before any release counts as reacquired.
        outcome = "run ended"
        for later in rows[index:]:
            decision = later["decision"]
            if decision["state"] == "default" or decision.get("emit_release"):
                outcome = "released"
                break
            if decision["state"] == "tracking":
                outcome = "reacquired"
                break
        low = int(np.searchsorted(truth_times, times[first] - 0.4))
        high = max(low + 1, int(np.searchsorted(truth_times, times[last] - 0.4)))
        span = [row for _, row in truth[low:high + 1]]
        distance = min(
            float(np.hypot(float(r["presenter_x"]) - PILLAR_SPOT[0], float(r["presenter_z"]) - PILLAR_SPOT[1])) for r in span
        )
        hidden = float(np.mean([r["marker_occluded_corners"] == "4" for r in span]))
        episodes.append({
            "at_seconds": float(times[first]),
            "seconds": duration,
            "outcome": outcome,
            "state_after": state,
            "at_pillar": bool(distance <= PILLAR_RADIUS_M),
            "presenter_xz": [float(span[0]["presenter_x"]), float(span[0]["presenter_z"])],
            "hfov": float(span[0]["hfov_deg"]),
            "fully_occluded_share": hidden,
            "in_view_share": float(np.mean([r["presenter_in_view"] == "1" for r in span])),
            "pan_change_deg": float(span[-1]["pan_deg"]) - float(span[0]["pan_deg"]),
            "error_x_at_end": float(span[-1]["err_x"]) if span[-1]["err_x"] else None,
        })
    return episodes


def unity_response_time(rows):
    """Seconds from a pan command arriving (camera stopped) to 90% of the steady pan speed."""
    time = np.array([float(row["time"]) for row in rows])
    pan = np.array([float(row["pan_deg"]) for row in rows])
    pan = np.unwrap(np.radians(pan)) * 180 / np.pi
    commands = [(float(row["time"]), row["cmd_dir"], row["cmd_step"]) for row in rows if row["cmd_dir"]]
    runs = []
    for at, direction, step in commands:
        if runs and runs[-1][1] == direction and runs[-1][2] == step:
            continue
        runs.append((at, direction, step))
    rises, delays = [], []
    for index in range(1, len(runs) - 1):
        at, direction, step = runs[index]
        end = runs[index + 1][0]
        if direction not in ("left", "right") or runs[index - 1][1] not in ("stop", "watchdog_stop") or end - at < 0.8:
            continue
        window = (time >= at) & (time <= end)
        t, angle = time[window], pan[window]
        if len(t) < 40:
            continue
        step_size = max(1, int(0.05 / np.median(np.diff(t))))
        speed = np.abs(angle[step_size:] - angle[:-step_size]) / (t[step_size:] - t[:-step_size])
        ts = t[step_size:]
        steady = float(np.median(speed[ts >= at + 0.6]))
        if steady < 0.3:
            continue
        t10 = next((x - at for x, v in zip(ts, speed) if v >= 0.1 * steady), None)
        t90 = next((x - at for x, v in zip(ts, speed) if v >= 0.9 * steady), None)
        if t10 is not None and t90 is not None:
            delays.append(t10)
            rises.append(t90 - t10)
    if not rises:
        return {"events": 0}
    return {"events": len(rises), "delay_seconds_median": float(np.median(delays)),
            "response_seconds_median": float(np.median(rises)),
            "response_seconds_quartiles": [float(np.percentile(rises, 25)), float(np.percentile(rises, 75))]}


def unity_metrics(name):
    directory = ROOT / "unity_metrics"
    with (directory / f"{name}_frames.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    time = np.array([float(row["time"]) for row in rows])
    dt = np.array([float(row["dt"]) for row in rows])
    error = np.array([float(row["err_x"]) if row["err_x"] else np.nan for row in rows])
    error_y = np.array([float(row["err_y"]) if row["err_y"] else np.nan for row in rows])
    in_view = np.array([row["presenter_in_view"] == "1" for row in rows])
    commands = [(float(row["time"]), row["cmd_dir"], row["cmd_step"]) for row in rows if row["cmd_dir"]]
    moves = [command for command in commands if command[1] in MOVES]
    if not moves:
        return {"valid": False}
    window = time >= moves[0][0]
    weight = dt[window]
    absolute = np.abs(error[window])
    visible = in_view[window]
    total = float(weight.sum())
    keep = lambda limit: float(weight[visible & (absolute <= limit)].sum() / total)
    order = np.argsort(absolute)
    cumulative = np.cumsum(weight[order]) / total
    quantile = lambda q: float(absolute[order][np.searchsorted(cumulative, q)])
    episodes = int(np.sum(~visible[1:] & visible[:-1]) + (0 if visible[0] else 1))
    # Split the exits: sideways (pan could not keep up) or only off the top/bottom edge.
    sideways = absolute > 0.5
    vertical_only = ~visible & ~sideways
    count = lambda flags: int(np.sum(flags[1:] & ~flags[:-1]) + (1 if flags[0] else 0))
    exits = {"sideways_exit_count": count(sideways), "sideways_exit_seconds": float(weight[sideways].sum()),
             "vertical_only_exit_count": count(vertical_only),
             "vertical_only_exit_seconds": float(weight[vertical_only].sum()),
             "keep15_x_only": float(weight[absolute <= 0.15].sum() / total),
             "y_abs_p90": float(np.nanpercentile(np.abs(error_y[window]), 90))}
    directions = [direction for _, direction, _ in moves]
    reversals = sum(1 for a, b in zip(directions, directions[1:]) if is_reversal(a, b))
    steps = np.array([int(step) for _, direction, step in moves if "left" in direction or "right" in direction])
    return {"valid": True, "steering_from_seconds": moves[0][0], "window_seconds": total,
            "out_of_view_count": episodes, "out_of_view_seconds": float(weight[~visible].sum()),
            "keep10": keep(0.10), "keep15": keep(0.15),
            "x_median": quantile(0.5), "x_p90": quantile(0.9), "x_max": float(np.nanmax(absolute)),
            "reversals_per_minute": reversals / total * 60,
            "pan_commands_level_5_plus": float((steps >= 5).mean()) if len(steps) else 0.0,
            "watchdog_stops": sum(1 for _, direction, _ in commands if direction == "watchdog_stop"), **exits,
            **_reversal_and_split(rows, moves, total), "response": unity_response_time(rows)}


def _reversal_and_split(rows, moves, total):
    classes = classify_reversals(rows, moves, total)
    events = classes.pop("events")
    return {"reversal_classes_per_minute": classes,
            "by_presenter_state": split_by_presenter_state(rows, moves[0][0], events)}


def classify_exits(name):
    """Count and time out-of-view episodes by cause.

    release: overlaps a period in which the tracker had lost and released the
      target (from the last frame with its box until tracking resumes).
    near_vertical: the target point left only through the top or bottom edge
      while the presenter's full box was taller than the frame (close to the camera).
    other: everything else.
    """
    rows, times, truth = load_run(name)
    released = [i for i, row in enumerate(rows) if row["decision"]["reason"] == "target_released_after_grace"]
    intervals = []
    for index in released:
        last_seen = index
        while last_seen > 0 and target_box(rows[last_seen]) is None:
            last_seen -= 1
        resumed = index
        while resumed < len(rows) - 1 and rows[resumed]["decision"]["state"] != "tracking":
            resumed += 1
        intervals.append((times[last_seen] - 0.5, times[resumed]))
    first_move = next((t for t, row in truth if row["cmd_dir"] in MOVES), None)
    causes = {key: {"count": 0, "seconds": 0.0} for key in ("near_vertical", "release", "other")}
    episode = []
    def close():
        if not episode:
            return
        start, end = episode[0][0], episode[-1][0]
        seconds = sum(float(row["dt"]) for _, row in episode)
        sideways = any(abs(float(row["err_x"])) > 0.5 for _, row in episode if row["err_x"])
        height = np.median([float(row["box_y1"]) - float(row["box_y0"]) for _, row in episode])
        frame_height = 360.0
        if any(start <= b and end >= a for a, b in intervals):
            cause = "release"
        elif not sideways and height >= frame_height:
            cause = "near_vertical"
        else:
            cause = "other"
        causes[cause]["count"] += 1
        causes[cause]["seconds"] += seconds
        episode.clear()
    for t, row in truth:
        if first_move is None or t < first_move:
            continue
        if row["presenter_in_view"] == "1":
            close()
        else:
            episode.append((t, row))
    close()
    return causes


def verdict(unity, tracker):
    return {"lost": unity["out_of_view_count"] == LIMITS["lost"] and not tracker["lost_at_seconds"],
            "keep15": unity["keep15"] >= LIMITS["keep15_min"],
            "x_max": unity["x_max"] < LIMITS["x_max_below"],
            "reversals": unity["reversal_classes_per_minute"]["hunting"] <= LIMITS["hunting_per_minute_max"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("prefix")
    parser.add_argument("labels", nargs="+")
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args(argv)
    output = []
    for label in args.labels:
        for repeat in range(1, args.repeats + 1):
            name = f"{args.prefix}_{label}_{repeat}"
            directory = ROOT / "unity_metrics"
            if not (directory / f"{name}_frames.csv").exists():
                output.append({"name": name, "label": label, "missing": True})
                continue
            meta = json.loads((directory / f"{name}_meta.json").read_text())
            tracker = metrics(name)
            unity = unity_metrics(name)
            if unity.get("valid"):
                unity["exit_causes"] = classify_exits(name)
                unity.update(registrations_and_releases(name))
                unity["grace_episodes"] = grace_episodes(name)
            entry = {"name": name, "label": label, "run_id": meta["run_id"],
                     "marker_mode": meta["status_before"].get("marker_mode"),
                     "unity_summary": json.loads((directory / f"{name}_summary.json").read_text()),
                     "unity": unity, "tracker": tracker,
                     "valid": bool(tracker.get("valid") and unity.get("valid"))}
            if entry["valid"]:
                entry["pass"] = verdict(unity, tracker)
            output.append(entry)
    print(json.dumps(output, indent=1))


if __name__ == "__main__":
    main()
