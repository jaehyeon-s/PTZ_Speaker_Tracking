"""Ground-truth metrics for Unity scenario runs.

Run: python tools/scenario_metrics.py NAME [NAME ...] [--markdown OUT.md] [--json OUT.json]
     python tools/scenario_metrics.py --prefix PREFIX          # every recorded run of a series

Reads, per run, test_video/unity_metrics/NAME_{meta.json,frames.csv,truth.json,actors.csv}
and the tracker's test_video/traces/NAME.jsonl and test_video/logs/NAME.log.

Ground truth
- truth.json: the presenter timetable. Each segment names who should be
  framed: "A", "A|B" (handover in progress, either counts) or none. Segments
  without a presenter are left out of every metric.
- actors.csv: every person's box projected into the camera picture, per
  Unity frame, in the same 640x360 pixels as the tracker's frames.

Metrics (seconds are Unity seconds; only time with a presenter counts)
- centered_ratio: share of time a correct presenter's framing point (box
  centre x, 35 % from the top) is within --center-tolerance of the picture
  centre (as a fraction of the width), i.e. the camera result.
- wrong_seconds: time the tracker's target box matched (IoU >= --match-iou)
  a person who was not the presenter then.
- switch latency: from a presenter's present_start to the camera keeping that
  person centred for --hold-seconds. Reported for handovers (after another
  presenter) and re-entries (after a time without presenter).
- flips: after a handover to B, how often the camera went back from B to A.
- releases: tracker releases while a presenter was present.

The tracker trace is placed on the Unity time line from the log clock and
then shifted by the offset (searched within +-2 s) at which the tracker's
person boxes best overlap the actors' boxes; this also absorbs the video delay.
"""
from __future__ import annotations

import argparse
import bisect
import csv
import datetime
import json
import math
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "test_video"
FIRST_STATUS = re.compile(r"(\S+ \S+),(\d+) INFO Tracking status: fps=0.0")
FRAME_WIDTH, FRAME_HEIGHT = 640.0, 360.0

# Success criterion per scenario, fixed (--criterion NAME=KIND overrides one).
SCENARIO_KINDS = {
    "handover_adjacent": "handover",
    "handover_then_exit": "handover",
    "cross_front": "no_wrong",
    "cross_behind": "no_wrong",
    "walk_alongside": "no_wrong",
    "walk_alongside_back": "no_wrong",
    "audience_question": "no_wrong",
    "board_writing": "no_release",
    "exit_reenter": "reentry",
    "lecture_front": "classic",
    "walk_audience": "classic",
}
HANDOVER_LIMIT_SECONDS = 3.0
REENTRY_LIMIT_SECONDS = 5.0  # provisional
CRITERIA_TEXT = {
    "handover": f"B의 present_start 뒤 {HANDOVER_LIMIT_SECONDS:g}초 안에 전환, A·B 왕복 없음",
    "no_wrong": "엉뚱한 사람 추종 0초",
    "no_release": "대상 해제 없음",
    "reentry": f"재입장 뒤 다시 잡기까지 {REENTRY_LIMIT_SECONDS:g}초 이내(임시)",
    "classic": "기존 네 기준(이탈 0, ±15% 유지 80% 이상, 최대 오차 0.35 미만, 헌팅 분당 7회 이하)",
    "none": "기준 없음",
}


def scenario_kind(scenario: str, overrides: dict[str, str] | None = None) -> str:
    if overrides and scenario in overrides:
        return overrides[scenario]
    return SCENARIO_KINDS.get(scenario, "none")


# --------------------------------------------------------------------------
# Ground truth parsing


@dataclass(frozen=True)
class Segment:
    start: float
    end: float
    actors: frozenset[str] | None  # None: nobody to frame


@dataclass
class Truth:
    segments: list[Segment]
    present_starts: list[tuple[str, float]] = field(default_factory=list)

    def at(self, t: float) -> frozenset[str] | None:
        for segment in self.segments:
            if segment.start <= t < segment.end:
                return segment.actors
        return None


_START_KEYS = ("start", "t0", "start_sec", "start_time", "from", "begin", "time")
_END_KEYS = ("end", "t1", "end_sec", "end_time", "to", "until")
_WHO_KEYS = ("speaker", "presenter", "truth", "who", "actor", "actors", "value", "label")
_ID_KEYS = ("id", "name", "actor", "presenter", "speaker", "label")


def _first(item: dict, keys):
    for key in keys:
        if key in item and item[key] is not None:
            return item[key]
    return None


def _actors(value) -> frozenset[str] | None:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        names = [str(v).strip() for v in value]
    else:
        names = [part.strip() for part in str(value).split("|")]
    names = [name for name in names if name and name.lower() not in {"none", "null", "-"}]
    return frozenset(names) or None


def parse_truth(data) -> Truth:
    raw = data
    if isinstance(data, dict):
        raw = next(
            (data[key] for key in ("segments", "timeline", "truth", "schedule", "speakers", "presenters")
             if isinstance(data.get(key), list) and data[key] and isinstance(data[key][0], dict)
             and _first(data[key][0], _START_KEYS) is not None),
            None,
        )
    if not isinstance(raw, list):
        raise ValueError("truth.json: no list of segments with a start time found")
    items = sorted(
        (item for item in raw if isinstance(item, dict) and _first(item, _START_KEYS) is not None),
        key=lambda item: float(_first(item, _START_KEYS)),
    )
    segments = []
    for index, item in enumerate(items):
        start = float(_first(item, _START_KEYS))
        end = _first(item, _END_KEYS)
        if end is None:
            end = float(_first(items[index + 1], _START_KEYS)) if index + 1 < len(items) else math.inf
        actors = _actors(_first(item, _WHO_KEYS))
        if item.get("evaluate") is False or str(item.get("type", "")).lower() == "none":
            actors = None  # left out of the evaluation
        segments.append(Segment(start, float(end), actors))
    present = []
    _collect_present_starts(data, present)
    return Truth(segments, sorted(set(present), key=lambda item: item[1]))


def _collect_present_starts(node, out: list[tuple[str, float]], owner: str | None = None) -> None:
    """Explicit present_start values anywhere in truth.json (single values or lists)."""
    if isinstance(node, dict):
        if node.get("type") == "present_start" and isinstance(node.get("t"), (int, float)) and node.get("actor"):
            # Unity events: {"t": 12.3, "actor": "B", "type": "present_start"}
            out.append((str(node["actor"]), float(node["t"])))
            return
        who = _first(node, _ID_KEYS)
        name = str(who) if isinstance(who, (str, int)) and "|" not in str(who) else owner
        for key, value in node.items():
            if key in ("present_start", "present_starts"):
                values = value if isinstance(value, list) else [value]
                for item in values:
                    if isinstance(item, (int, float)) and name:
                        out.append((name, float(item)))
            elif isinstance(value, (dict, list)):
                # A dict keyed by actor name, e.g. {"B": {"present_start": 12}}.
                child_owner = key if isinstance(value, dict) and _first(value, _ID_KEYS) is None else name
                _collect_present_starts(value, out, child_owner)
    elif isinstance(node, list):
        for item in node:
            _collect_present_starts(item, out, owner)


@dataclass(frozen=True)
class PresenceEvent:
    actor: str
    start: float
    end: float  # end of this contiguous presence
    kind: str  # initial, handover, reentry
    previous: frozenset[str]


def presence_events(truth: Truth) -> list[PresenceEvent]:
    """Each time a person starts to be the (or a) presenter."""
    events = []
    previous: frozenset[str] = frozenset()
    seen_any = False
    for index, segment in enumerate(truth.segments):
        current = segment.actors or frozenset()
        for actor in sorted(current - previous):
            end = segment.end
            for later in truth.segments[index + 1:]:
                if later.actors and actor in later.actors and later.start <= end + 1e-6:
                    end = later.end
                else:
                    break
            if not seen_any:
                kind = "initial"
            elif previous:
                kind = "handover"
            else:
                kind = "reentry"
            events.append(PresenceEvent(actor, segment.start, end, kind, previous))
        if current:
            seen_any = True
        previous = current
    explicit = {actor: [t for a, t in truth.present_starts if a == actor] for actor, _ in truth.present_starts}
    if explicit:
        # present_start from truth.json wins over the segment boundary.
        adjusted = []
        for event in events:
            # A handover segment starts when the handover starts; the receiver's
            # present_start comes later inside it.
            times = [t for t in explicit.get(event.actor, []) if event.start - 1.0 <= t < event.end]
            start = min(times) if times else event.start
            adjusted.append(PresenceEvent(event.actor, start, event.end, event.kind, event.previous))
        events = adjusted
    return events


# --------------------------------------------------------------------------
# actors.csv


def _column(fields: list[str], candidates, required=True, what=""):
    lowered = {name.lower(): name for name in fields}
    for candidate in candidates:
        if candidate in lowered:
            return lowered[candidate]
    if required:
        raise ValueError(f"actors.csv: no {what} column among {fields}")
    return None


def _utc_epoch(text: str) -> float:
    return datetime.datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()


class Actors:
    """Per-actor projected boxes on the Unity time line."""

    def __init__(self, rows: list[dict], utc_to_time=None, frame_to_time: dict | None = None) -> None:
        if not rows:
            raise ValueError("actors.csv is empty")
        fields = list(rows[0])
        time_column = _column(fields, ("time", "t", "elapsed", "elapsed_sec", "sec", "seconds"), required=False)
        utc_column = _column(fields, ("utc",), required=False)
        frame_column = _column(fields, ("frame",), required=False)
        if (
            time_column is None
            and (utc_column is None or utc_to_time is None)
            and (frame_column is None or not frame_to_time)
        ):
            raise ValueError(f"actors.csv: no time column among {fields}")
        actor_column = _column(fields, ("actor", "actor_id", "id", "name", "person", "role", "label"), what="actor")
        box_columns = None
        for names in (("box_x0", "box_y0", "box_x1", "box_y1"), ("x0", "y0", "x1", "y1"),
                      ("left", "top", "right", "bottom"), ("xmin", "ymin", "xmax", "ymax"),
                      ("x_min", "y_min", "x_max", "y_max")):
            if all(name in {f.lower() for f in fields} for name in names):
                box_columns = [_column(fields, (name,)) for name in names]
                break
        if box_columns is None:
            raise ValueError(f"actors.csv: no box columns among {fields}")
        visible_column = _column(fields, ("in_view", "visible", "box_in_view", "presenter_in_view", "on_screen"),
                                 required=False)
        self.tracks: dict[str, tuple[list[float], list[tuple[float, float, float, float] | None]]] = {}
        for row in rows:
            try:
                if time_column is not None:
                    t = float(row[time_column])
                elif frame_column is not None and frame_to_time:
                    # Unity frame number; frames.csv gives its run time.
                    frame = int(float(row[frame_column]))
                    if frame not in frame_to_time:
                        continue
                    t = frame_to_time[frame]
                else:
                    t = utc_to_time(_utc_epoch(row[utc_column]))
                box = tuple(float(row[c]) for c in box_columns)
            except (TypeError, ValueError):
                continue
            if visible_column is not None and str(row[visible_column]).strip() in {"0", "false", "False"}:
                box = None
            times, boxes = self.tracks.setdefault(str(row[actor_column]).strip(), ([], []))
            times.append(t)
            boxes.append(box)
        for times, boxes in self.tracks.values():
            order = sorted(range(len(times)), key=times.__getitem__)
            times[:] = [times[i] for i in order]
            boxes[:] = [boxes[i] for i in order]
        self.times = sorted({t for times, _ in self.tracks.values() for t in times})

    def boxes_at(self, t: float, max_gap: float = 0.15) -> dict[str, tuple[float, float, float, float]]:
        result = {}
        for actor, (times, boxes) in self.tracks.items():
            index = bisect.bisect_left(times, t)
            best = None
            for candidate in (index - 1, index):
                if 0 <= candidate < len(times) and abs(times[candidate] - t) <= max_gap:
                    if best is None or abs(times[candidate] - t) < abs(times[best] - t):
                        best = candidate
            if best is not None and boxes[best] is not None:
                result[actor] = boxes[best]
        return result


def framing_error(box) -> tuple[float, float]:
    """Framing point (centre x, 35 % from the top) relative to the picture centre."""
    x = (box[0] + box[2]) / 2
    y = box[1] + 0.35 * (box[3] - box[1])
    return (x - FRAME_WIDTH / 2) / FRAME_WIDTH, (y - FRAME_HEIGHT / 2) / FRAME_HEIGHT


def centered_actors(boxes: dict, tolerance: float) -> list[str]:
    """Actors whose framing point is inside the picture and near the centre, closest first."""
    found = []
    for actor, box in boxes.items():
        error_x, error_y = framing_error(box)
        if abs(error_x) <= tolerance and abs(error_y) <= 0.5:
            found.append((abs(error_x), actor))
    return [actor for _, actor in sorted(found)]


def iou(first, second) -> float:
    x1, y1 = max(first[0], second[0]), max(first[1], second[1])
    x2, y2 = min(first[2], second[2]), min(first[3], second[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = (first[2] - first[0]) * (first[3] - first[1]) + (second[2] - second[0]) * (second[3] - second[1]) - inter
    return inter / union if union > 0 else 0.0


def matched_actor(box, boxes: dict, minimum: float) -> str | None:
    best, best_iou = None, minimum
    for actor, actor_box in boxes.items():
        overlap = iou(box, actor_box)
        if overlap >= best_iou:
            best, best_iou = actor, overlap
    return best


# --------------------------------------------------------------------------
# Tracker trace on the Unity time line


def target_box(row) -> list[float] | None:
    target = row["decision"].get("target_track_id")
    if target is None:
        return None
    for detection in row["detections"]:
        if detection["label"] == "person" and detection["track_id"] == target:
            return detection["box"]
    return None


def align_offset(trace: list[dict], actors: Actors, estimate: float, window: float = 2.0,
                 step: float = 0.02) -> tuple[float, float]:
    """Offset (Unity time = trace time + offset) at which tracker boxes best overlap actor boxes."""
    start = trace[0]["timestamp"]
    samples = [
        (row["timestamp"] - start, [d["box"] for d in row["detections"] if d["label"] == "person"])
        for row in trace[:: max(1, len(trace) // 400)]
    ]
    samples = [sample for sample in samples if sample[1]]
    if not samples:
        return estimate, 0.0
    best = (estimate, -1.0)
    steps = int(round(window / step))
    for k in range(-steps, steps + 1):
        offset = estimate + k * step
        total = count = 0
        for t, boxes in samples:
            # Per actor, the best tracker box: the audience is not in actors.csv,
            # so tracker boxes without an actor must not count.
            for actor_box in actors.boxes_at(t + offset).values():
                if not -0.5 <= framing_error(actor_box)[0] <= 0.5:
                    continue  # outside the picture
                total += max(iou(box, actor_box) for box in boxes)
                count += 1
        score = total / count if count else 0.0
        if score > best[1] + 1e-9:
            best = (offset, score)
    return best


def _dt(times: list[float], index: int, cap: float = 0.2) -> float:
    if index + 1 < len(times):
        return max(0.0, min(cap, times[index + 1] - times[index]))
    return 0.0


def _first_sustained(times: list[float], flags: list[bool], start: float, end: float, hold: float) -> float | None:
    """First time in [start, end) from which flags stay true for `hold` seconds."""
    since = None
    for t, flag in zip(times, flags):
        if t < start:
            continue
        if t >= end and since is None:
            return None
        if flag:
            since = t if since is None else since
            if t - since >= hold:
                return since
        else:
            since = None
    return None


def debounced(times: list[float], labels: list[str | None], hold: float) -> list[tuple[float, str]]:
    """Runs of the same label that last at least `hold` seconds (None runs ignored)."""
    runs: list[tuple[float, str]] = []
    current, since = None, None
    for t, label in zip(times + [math.inf], labels + [None]):
        if label != current:
            if current is not None and since is not None and t - since >= hold:
                if not runs or runs[-1][1] != current:
                    runs.append((since, current))
            current, since = label, t
    return runs


def readable_unregistered_seconds(truth: Truth, readable: list[tuple[float, bool]] | None,
                                  trace: list[dict], trace_times: list[float]) -> float | None:
    """Time a readable marker was in the picture while the tracker held nobody.

    `readable` is Unity's marker_readable per frame; None if the simulator did
    not record it. Only time with a presenter counts.
    """
    if readable is None:
        return None
    held = [row["decision"].get("target_track_id") is not None
            and row["decision"].get("state") not in {"default", "confirming"} for row in trace]
    total = 0.0
    times = [t for t, _ in readable]
    for index, (t, is_readable) in enumerate(readable):
        if not is_readable or not truth.at(t):
            continue
        position = bisect.bisect_right(trace_times, t) - 1
        if position < 0 or not held[position]:
            total += _dt(times, index)
    return round(total, 2)


def evaluate(truth: Truth, actors: Actors, trace: list[dict], offset: float, *, kind: str,
             tolerance: float = 0.15, match_iou: float = 0.3, hold: float = 0.5,
             handover_limit: float = 3.0, reentry_limit: float | None = None,
             readable: list[tuple[float, bool]] | None = None) -> dict:
    times = actors.times
    boxes_by_time = [actors.boxes_at(t) for t in times]
    centred = [centered_actors(boxes, tolerance) for boxes in boxes_by_time]
    evaluated = centred_correct = 0.0
    for index, t in enumerate(times):
        present = truth.at(t)
        if not present:
            continue
        dt = _dt(times, index)
        evaluated += dt
        if any(actor in present for actor in centred[index]):
            centred_correct += dt

    trace_start = trace[0]["timestamp"] if trace else 0.0
    trace_times = [row["timestamp"] - trace_start + offset for row in trace]
    wrong = 0.0
    wrong_spans: list[tuple[float, str]] = []
    releases = 0
    tracked_actor: list[str | None] = []
    followed_actor: list[str | None] = []
    for index, row in enumerate(trace):
        t = trace_times[index]
        present = truth.at(t)
        box = target_box(row)
        actor = matched_actor(box, actors.boxes_at(t), match_iou) if box is not None else None
        tracked_actor.append(actor)
        # While a marker holder is only being confirmed the camera does not
        # follow it, so that is not following anyone.
        following = row["decision"].get("state") not in {"default", "confirming"}
        followed_actor.append(actor if following else None)
        if not present:
            continue
        if row["decision"].get("emit_release"):
            releases += 1
        if following and actor is not None and actor not in present:
            wrong += _dt(trace_times, index)
            if not wrong_spans or wrong_spans[-1][1] != actor or t - wrong_spans[-1][0] > 1.0:
                wrong_spans.append((t, actor))

    on_camera = [names[0] if names else None for names in centred]
    # The tracker's followed person at each simulator time (latest trace row).
    followed_at_unity = []
    position = 0
    for t in times:
        while position + 1 < len(trace_times) and trace_times[position + 1] <= t:
            position += 1
        followed_at_unity.append(
            followed_actor[position] if trace_times and trace_times[position] <= t else None
        )
    events = []
    for event in presence_events(truth):
        flags = [event.actor in names for names in centred]
        reached = _first_sustained(times, flags, event.start, event.end, hold)
        tracker_flags = [label == event.actor for label in tracked_actor]
        tracker_reached = _first_sustained(trace_times, tracker_flags, event.start, event.end, hold)
        # Centred on the camera only counts once the tracker follows that
        # person too: a still camera can have someone walk into its centre
        # (exit_reenter in full1: centred at once, taken 9.8 s later).
        acquired_flags = [flag and followed == event.actor for flag, followed in zip(flags, followed_at_unity)]
        acquired = _first_sustained(times, acquired_flags, event.start, event.end, hold)
        entry = {"actor": event.actor, "kind": event.kind, "present_start": round(event.start, 2),
                 "camera_latency": None if reached is None else round(reached - event.start, 2),
                 "tracker_latency": None if tracker_reached is None else round(tracker_reached - event.start, 2),
                 "acquired_latency": None if acquired is None else round(acquired - event.start, 2)}
        if event.kind == "handover":
            others = set(event.previous) - {event.actor}
            window = [i for i, t in enumerate(times) if event.start <= t < event.end]
            runs = debounced([times[i] for i in window],
                             [on_camera[i] if on_camera[i] in others | {event.actor} else None for i in window],
                             hold)
            reached_new = False
            flips = 0
            for _, label in runs:
                if label == event.actor:
                    reached_new = True
                elif reached_new and label in others:
                    flips += 1
            entry["flips"] = flips
        events.append(entry)

    registered_at = next(
        (trace_times[i] for i, row in enumerate(trace)
         if row["decision"].get("emit_handoff") or row["decision"].get("speaker_id") is not None),
        None,
    )
    handovers = [e for e in events if e["kind"] == "handover"]
    reentries = [e for e in events if e["kind"] == "reentry"]
    result = {
        "evaluated_seconds": round(evaluated, 2),
        "centered_ratio": round(centred_correct / evaluated, 4) if evaluated else None,
        "wrong_seconds": round(wrong, 2),
        "wrong_episodes": [{"t": round(t, 2), "actor": a} for t, a in wrong_spans],
        "releases": releases,
        "events": events,
        "handover_latency": [e["acquired_latency"] for e in handovers],
        "flips": sum(e.get("flips", 0) for e in handovers),
        "reentry_latency": [e["acquired_latency"] for e in reentries],
        "kind": kind,
        "criterion": CRITERIA_TEXT[kind],
        "registered_at": None if registered_at is None else round(registered_at, 2),
        "readable_unregistered_seconds": readable_unregistered_seconds(truth, readable, trace, trace_times),
    }
    result["passed"] = judge(result, handover_limit, reentry_limit)
    if registered_at is None:
        # The tracker never took anyone: a still camera can look like a pass.
        result["passed"] = None
        result["invalid"] = "발표자 등록 없음"
    return result


def judge(result: dict, handover_limit: float, reentry_limit: float | None) -> bool | None:
    kind = result["kind"]
    if kind == "classic":
        checks = result.get("classic_checks")
        return None if not checks else all(checks.values())
    if kind == "handover":
        latencies = result["handover_latency"]
        if not latencies:
            return None
        return all(v is not None and v <= handover_limit for v in latencies) and result["flips"] == 0
    if kind == "no_wrong":
        return result["wrong_seconds"] == 0
    if kind == "no_release":
        return result["releases"] == 0
    if kind == "reentry":
        latencies = result["reentry_latency"]
        if not latencies:
            return None
        return all(v is not None and (reentry_limit is None or v <= reentry_limit) for v in latencies)
    return None


# --------------------------------------------------------------------------
# Files


def _log_start_epoch(name: str) -> float | None:
    path = ROOT / "logs" / f"{name}.log"
    if not path.exists():
        return None
    with path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            match = FIRST_STATUS.match(line)
            if match:
                stamp = datetime.datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S")
                return stamp.timestamp() + int(match.group(2)) / 1000
    return None


def load_and_evaluate(name: str, options) -> dict:
    metrics_dir = ROOT / "unity_metrics"
    meta_path = metrics_dir / f"{name}_meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    scenario = meta.get("scenario") or ""
    label = meta.get("label") or ""
    kind = scenario_kind(scenario, options.criterion_overrides)
    base = {"name": name, "scenario": scenario, "label": label, "kind": kind,
            "criterion": CRITERIA_TEXT[kind], "start_conditions": meta.get("start_conditions")}
    truth_path = metrics_dir / f"{name}_truth.json"
    actors_path = metrics_dir / f"{name}_actors.csv"
    if not truth_path.exists() or not actors_path.exists():
        return {**base, "error": "truth.json or actors.csv missing"}
    with (metrics_dir / f"{name}_frames.csv").open(newline="") as handle:
        frames = list(csv.DictReader(handle))
    pairs = [(float(row["time"]), _utc_epoch(row["utc"])) for row in frames if row.get("time") and row.get("utc")]
    run_epoch = pairs[0][1] - pairs[0][0] if pairs else None
    frame_to_time = {
        int(float(row["frame"])): float(row["time"]) for row in frames if row.get("frame") and row.get("time")
    }
    with actors_path.open(newline="") as handle:
        actors = Actors(list(csv.DictReader(handle)),
                        utc_to_time=(lambda epoch: epoch - run_epoch) if run_epoch is not None else None,
                        frame_to_time=frame_to_time)
    truth = parse_truth(json.loads(truth_path.read_text()))
    with (ROOT / "traces" / f"{name}.jsonl").open(encoding="utf-8") as handle:
        trace = [json.loads(line) for line in handle if line.strip()]
    log_epoch = _log_start_epoch(name)
    estimate = log_epoch - run_epoch if (log_epoch is not None and run_epoch is not None) else 0.0
    window = 2.0 if log_epoch is not None else 10.0
    offset, overlap = align_offset(trace, actors, estimate, window=window)
    readable = None
    if frames and "marker_readable" in frames[0]:
        readable = [(float(row["time"]), str(row["marker_readable"]).strip() in {"1", "true", "True"})
                    for row in frames if row.get("time")]
    result = evaluate(truth, actors, trace, offset, kind=kind, tolerance=options.center_tolerance,
                      match_iou=options.match_iou, hold=options.hold_seconds,
                      handover_limit=options.handover_limit, reentry_limit=options.reentry_limit,
                      readable=readable)
    if kind == "classic" and "invalid" not in result:
        result["classic_checks"] = classic_checks(name)
        result["passed"] = judge(result, options.handover_limit, options.reentry_limit)
    result["registration_marker_px"] = None
    if result["registered_at"] is not None:
        # Marker size in the simulator's frame closest to the registration.
        timed = [(float(row["time"]), row.get("marker_px")) for row in frames if row.get("time")]
        if timed:
            _, pixels = min(timed, key=lambda item: abs(item[0] - result["registered_at"]))
            result["registration_marker_px"] = float(pixels) if pixels not in (None, "") else None
    return {**base, **result, "offset_seconds": round(offset, 2), "offset_estimate": round(estimate, 2),
            "alignment_iou": round(overlap, 3)}


def classic_checks(name: str) -> dict | None:
    """The four earlier criteria (lecture_front, walk_audience) from compare_unity_series."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import compare_unity_series
    import tracker_run_metrics

    unity = compare_unity_series.unity_metrics(name)
    tracker = tracker_run_metrics.metrics(name, with_video=False)
    if not (unity.get("valid") and tracker.get("valid")):
        return None
    return compare_unity_series.verdict(unity, tracker)


def _fmt(value, digits=2):
    if value is None:
        return "-"
    if isinstance(value, list):
        return ", ".join(_fmt(v, digits) for v in value) if value else "-"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def markdown(results: list[dict]) -> str:
    lines = [
        "| 시나리오 | 조건 | 실행 | 등록(s) | 등록 시 마커(px) | 마커 읽힘·미등록(s) | 평가 시간(s) | 올바른 발표자 중심 비율 | 엉뚱한 사람 추종(s) | 전환 지연(s) | A·B 왕복 | 해제 | 재획득(s) | 성공 기준 | 판정 |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        if "error" in r:
            lines.append(f"| {r['scenario']} | {r['label']} | {r['name']} | - | - | - | - | - | - | - | - | - | - | {r['criterion']} | {r['error']} |")
            continue
        if r.get("invalid"):
            verdict = f"무효({r['invalid']})"
        else:
            verdict = {True: "통과", False: "실패", None: "판정 불가"}[r["passed"]]
        ratio = "-" if r["centered_ratio"] is None else f"{100 * r['centered_ratio']:.0f}%"
        lines.append(
            f"| {r['scenario']} | {r['label']} | {r['name']} | {_fmt(r.get('registered_at'), 1)} | "
            f"{_fmt(r.get('registration_marker_px'), 1)} | "
            f"{'기록 없음' if r.get('readable_unregistered_seconds') is None else _fmt(r['readable_unregistered_seconds'])} | "
            f"{_fmt(r['evaluated_seconds'], 1)} | {ratio} | "
            f"{_fmt(r['wrong_seconds'])} | {_fmt(r['handover_latency'])} | {r['flips']} | {r['releases']} | "
            f"{_fmt(r['reentry_latency'])} | {r['criterion']} | {verdict} |"
        )
    groups: dict[tuple[str, str], list[dict]] = {}
    for r in results:
        groups.setdefault((r["scenario"], r["label"]), []).append(r)
    lines += ["", "| 시나리오 | 조건 | 성공 기준 | 통과 | 무효 | 판정 불가·오류 |", "|---|---|---|---|---|---|"]
    for (scenario, label), runs in groups.items():
        passed = sum(1 for r in runs if r.get("passed") is True)
        invalid = sum(1 for r in runs if r.get("invalid"))
        undecided = sum(1 for r in runs if not r.get("invalid") and (r.get("passed") is None or "error" in r))
        valid = len(runs) - invalid
        lines.append(f"| {scenario} | {label} | {runs[0]['criterion']} | {passed}/{valid} | {invalid} | {undecided} |")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("names", nargs="*")
    parser.add_argument("--prefix", help="Evaluate every recorded run whose name starts with PREFIX_")
    parser.add_argument("--center-tolerance", type=float, default=0.15)
    parser.add_argument("--match-iou", type=float, default=0.3)
    parser.add_argument("--hold-seconds", type=float, default=0.5)
    parser.add_argument("--handover-limit", type=float, default=HANDOVER_LIMIT_SECONDS,
                        help="Seconds from B's present_start to B centred (handover scenarios)")
    parser.add_argument("--reentry-limit", type=float, default=REENTRY_LIMIT_SECONDS,
                        help="Pass limit for the re-acquisition time (provisional)")
    parser.add_argument("--criterion", action="append", default=[], metavar="SCENARIO=KIND",
                        help=f"Override the scenario kind ({', '.join(CRITERIA_TEXT)})")
    parser.add_argument("--markdown", type=Path)
    parser.add_argument("--json", type=Path)
    options = parser.parse_args(argv)
    options.criterion_overrides = {}
    for item in options.criterion:
        scenario, _, kind = item.partition("=")
        if kind not in CRITERIA_TEXT:
            parser.error(f"--criterion {item}: kind must be one of {', '.join(CRITERIA_TEXT)}")
        options.criterion_overrides[scenario] = kind
    names = list(options.names)
    if options.prefix:
        names += sorted(
            path.name[: -len("_meta.json")]
            for path in (ROOT / "unity_metrics").glob(f"{options.prefix}_*_meta.json")
        )
    if not names:
        parser.error("give run names or --prefix")
    results = [load_and_evaluate(name, options) for name in names]
    results.sort(key=lambda r: (r["scenario"], r["label"], r["name"]))
    text = markdown(results)
    if options.markdown:
        options.markdown.write_text(text, encoding="utf-8")
    if options.json:
        options.json.write_text(json.dumps(results, indent=1, ensure_ascii=False), encoding="utf-8")
    sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
