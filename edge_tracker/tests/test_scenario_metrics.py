from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))

import scenario_metrics as sm  # noqa: E402

CENTRE = (290.0, 100.0, 350.0, 300.0)  # framing point at x=320
LEFT = (40.0, 100.0, 100.0, 300.0)
RIGHT = (540.0, 100.0, 600.0, 300.0)


def actors_rows(script, step=0.05, end=20.0):
    """script(t) -> {actor: box or None}; one row per actor and Unity frame."""
    rows = []
    t = 0.0
    while t < end:
        for actor, box in script(t).items():
            row = {"time": f"{t:.3f}", "actor": actor}
            if box is None:
                row.update(x0="", y0="", x1="", y1="", in_view="0")
            else:
                row.update(x0=box[0], y0=box[1], x1=box[2], y1=box[3], in_view="1")
            rows.append(row)
        t += step
    return rows


def trace_rows(script, step=0.05, end=20.0, start=1000.0, releases=()):
    """script(t) -> list of (track_id, box) and the target track id."""
    rows = []
    t = 0.0
    while t < end:
        people, target = script(t)
        rows.append({
            "timestamp": start + t,
            "detections": [
                {"label": "person", "confidence": 0.9, "track_id": track, "box": list(box)}
                for track, box in people
            ],
            "decision": {"target_track_id": target, "speaker_id": 1 if target is not None else None,
                         "emit_release": any(abs(t - r) < step / 2 for r in releases)},
        })
        t += step
    return rows


class TruthParsingTest(unittest.TestCase):
    def test_segments_with_handover_and_none(self):
        truth = sm.parse_truth({"segments": [
            {"start": 0, "end": 2, "speaker": "none"},
            {"start": 2, "end": 8, "speaker": "A"},
            {"start": 8, "end": 9, "speaker": "A|B"},
            {"start": 9, "end": 15, "speaker": "B"},
            {"start": 15, "end": 17, "speaker": None},
            {"start": 17, "end": 20, "speaker": "B"},
        ]})
        self.assertIsNone(truth.at(1.0))
        self.assertEqual(truth.at(8.5), frozenset({"A", "B"}))
        events = sm.presence_events(truth)
        self.assertEqual([(e.actor, e.start, e.kind) for e in events],
                         [("A", 2.0, "initial"), ("B", 8.0, "handover"), ("B", 17.0, "reentry")])
        self.assertEqual(events[1].end, 15.0)

    def test_explicit_present_start_and_open_end(self):
        truth = sm.parse_truth({
            "timeline": [{"t0": 0, "presenter": "A"}, {"t0": 10, "presenter": "B"}],
            "actors": [{"id": "B", "present_start": 9.4}],
        })
        self.assertEqual(truth.segments[0].end, 10.0)
        events = sm.presence_events(truth)
        self.assertEqual(events[1].start, 9.4)

    def test_scenario_kinds_are_fixed(self):
        self.assertEqual(sm.scenario_kind("handover_then_exit"), "handover")
        self.assertEqual(sm.scenario_kind("cross_behind"), "no_wrong")
        self.assertEqual(sm.scenario_kind("walk_alongside_back"), "no_wrong")
        self.assertEqual(sm.scenario_kind("board_writing"), "no_release")
        self.assertEqual(sm.scenario_kind("exit_reenter"), "reentry")
        self.assertEqual(sm.scenario_kind("lecture_front"), "classic")
        self.assertEqual(sm.scenario_kind("crossing"), "none")
        self.assertEqual(sm.scenario_kind("crossing", {"crossing": "no_wrong"}), "no_wrong")

    def test_unity_truth_format(self):
        # Shape of the simulator's truth.json (run_0212 and the handover rules).
        truth = sm.parse_truth({
            "segments": [
                {"from": 0, "to": 10, "type": "presenter", "truth": "A", "actors": ["A"], "evaluate": True},
                {"from": 10, "to": 15.5, "type": "transition", "truth": "A|B", "actors": ["A", "B"],
                 "evaluate": True},
                {"from": 15.5, "to": 30, "type": "presenter", "truth": "B", "actors": ["B"], "evaluate": True},
                {"from": 30, "to": 40, "type": "none", "truth": "none", "actors": [], "evaluate": False},
            ],
            "events": [
                {"t": 0, "actor": "A", "type": "present_start", "label": "A presents"},
                {"t": 10, "actor": "A", "type": "handover", "label": "card to B"},
                {"t": 12.5, "actor": "B", "type": "present_start", "label": "B presents"},
            ],
        })
        self.assertIsNone(truth.at(35.0))
        events = sm.presence_events(truth)
        self.assertEqual([(e.actor, e.start, e.kind) for e in events],
                         [("A", 0.0, "initial"), ("B", 12.5, "handover")])

    def test_actors_csv_with_frame_numbers(self):
        rows = [{"frame": "9352", "actor": "A", "x0": "353.1", "y0": "125.2", "x1": "411.0", "y1": "321.0",
                 "in_view": "1", "state": "waiting", "has_marker": "1"}]
        actors = sm.Actors(rows, frame_to_time={9352: 0.0})
        self.assertEqual(actors.boxes_at(0.0)["A"], (353.1, 125.2, 411.0, 321.0))

    def test_actors_csv_reports_missing_columns(self):
        with self.assertRaises(ValueError) as context:
            sm.Actors([{"time": "0", "who": "A", "a": "1"}])
        self.assertIn("actor", str(context.exception))


class HandoverTest(unittest.TestCase):
    def setUp(self):
        self.truth = sm.parse_truth([
            {"start": 0, "end": 8, "speaker": "A"},
            {"start": 8, "end": 9, "speaker": "A|B"},
            {"start": 9, "end": 20, "speaker": "B"},
        ])

    def run_case(self, camera_on_b_at, back_to_a=None):
        def scene(t):
            on_b = t >= camera_on_b_at and not (back_to_a and back_to_a[0] <= t < back_to_a[1])
            return {"A": RIGHT if on_b else CENTRE, "B": CENTRE if on_b else LEFT}

        def tracker(t):
            boxes = scene(t)
            on_b = boxes["B"] == CENTRE
            return [(1, boxes["A"]), (2, boxes["B"])], 2 if on_b else 1

        actors = sm.Actors(actors_rows(scene))
        trace = trace_rows(tracker)
        return sm.evaluate(self.truth, actors, trace, offset=0.0, kind="handover")

    def test_quick_handover_passes(self):
        result = self.run_case(camera_on_b_at=10.0)
        self.assertAlmostEqual(result["handover_latency"][0], 2.0, places=1)
        self.assertEqual(result["flips"], 0)
        # Still on A for 1 s after the overlap (9-10 s): counted as following the
        # wrong person, but the handover criterion uses the latency instead.
        self.assertAlmostEqual(result["wrong_seconds"], 1.0, delta=0.06)
        self.assertTrue(result["passed"])

    def test_late_handover_fails_and_counts_following_a_as_wrong(self):
        result = self.run_case(camera_on_b_at=13.0)
        self.assertGreater(result["handover_latency"][0], 3.0)
        self.assertAlmostEqual(result["wrong_seconds"], 4.0, delta=0.15)
        self.assertFalse(result["passed"])

    def test_going_back_to_a_is_a_flip(self):
        result = self.run_case(camera_on_b_at=10.0, back_to_a=(12.0, 13.5))
        self.assertEqual(result["flips"], 1)
        self.assertFalse(result["passed"])

    def test_centered_ratio_counts_either_person_during_the_overlap(self):
        result = self.run_case(camera_on_b_at=10.0)
        # A centred until 9 s (correct: A, then A|B), gap 9-10 s, then B.
        self.assertAlmostEqual(result["centered_ratio"], 19.0 / 20.0, delta=0.01)


class OtherCriteriaTest(unittest.TestCase):
    def test_none_segments_are_left_out(self):
        truth = sm.parse_truth([{"start": 0, "end": 10, "speaker": "none"},
                                {"start": 10, "end": 20, "speaker": "A"}])
        actors = sm.Actors(actors_rows(lambda t: {"A": CENTRE, "C": LEFT}))
        # The tracker follows C while nobody presents, then A.
        trace = trace_rows(lambda t: ([(1, CENTRE), (3, LEFT)], 3 if t < 10 else 1), releases=(5.0,))
        result = sm.evaluate(truth, actors, trace, offset=0.0, kind="no_wrong")
        self.assertAlmostEqual(result["evaluated_seconds"], 10.0, delta=0.06)
        self.assertEqual(result["wrong_seconds"], 0.0)
        self.assertEqual(result["releases"], 0)
        self.assertTrue(result["passed"])

    def test_run_without_registration_is_invalid_not_passed(self):
        # cross_front in smoke1: no marker read, the camera stood still on A.
        truth = sm.parse_truth([{"start": 0, "end": 20, "speaker": "A"}])
        actors = sm.Actors(actors_rows(lambda t: {"A": CENTRE, "B": LEFT}))
        trace = trace_rows(lambda t: ([(1, CENTRE), (2, LEFT)], None))
        result = sm.evaluate(truth, actors, trace, offset=0.0, kind="no_wrong")
        self.assertEqual(result["wrong_seconds"], 0.0)
        self.assertIsNone(result["registered_at"])
        self.assertIsNone(result["passed"])
        self.assertEqual(result["invalid"], "발표자 등록 없음")
        table = sm.markdown([{**result, "scenario": "cross_front", "label": "n0", "name": "x",
                              "registration_marker_px": None}])
        self.assertIn("무효(발표자 등록 없음)", table)
        self.assertIn("기록 없음", table)

    def test_readable_marker_without_a_held_speaker_is_counted(self):
        truth = sm.parse_truth([{"start": 0, "end": 20, "speaker": "A"}])
        actors = sm.Actors(actors_rows(lambda t: {"A": CENTRE}))
        trace = trace_rows(lambda t: ([(1, CENTRE)], 1 if t >= 5 else None))
        for row in trace:
            row["decision"]["state"] = "tracking" if row["decision"]["target_track_id"] else "confirming"
        readable = [(k * 0.05, True) for k in range(400)]
        result = sm.evaluate(truth, actors, trace, offset=0.0, kind="no_wrong", readable=readable)
        self.assertAlmostEqual(result["readable_unregistered_seconds"], 5.0, delta=0.06)
        self.assertIsNone(sm.evaluate(truth, actors, trace, offset=0.0, kind="no_wrong")
                          ["readable_unregistered_seconds"])

    def test_a_candidate_being_confirmed_is_not_followed(self):
        truth = sm.parse_truth([{"start": 0, "end": 20, "speaker": "A"}])
        actors = sm.Actors(actors_rows(lambda t: {"A": CENTRE, "C": LEFT}))
        trace = trace_rows(lambda t: ([(1, CENTRE), (3, LEFT)], 3 if t < 2 else 1))
        for row in trace:
            row["decision"]["state"] = "confirming" if row["decision"]["target_track_id"] == 3 else "tracking"
        result = sm.evaluate(truth, actors, trace, offset=0.0, kind="no_wrong")
        self.assertEqual(result["wrong_seconds"], 0.0)

    def test_release_while_presenting_fails_board_scenario(self):
        truth = sm.parse_truth([{"start": 0, "end": 20, "speaker": "A"}])
        actors = sm.Actors(actors_rows(lambda t: {"A": CENTRE}))
        trace = trace_rows(lambda t: ([(1, CENTRE)], 1), releases=(7.0,))
        result = sm.evaluate(truth, actors, trace, offset=0.0, kind="no_release")
        self.assertEqual(result["releases"], 1)
        self.assertFalse(result["passed"])

    def test_reentry_time_is_measured_from_present_start(self):
        truth = sm.parse_truth([{"start": 0, "end": 6, "speaker": "A"},
                                {"start": 6, "end": 10, "speaker": "none"},
                                {"start": 10, "end": 20, "speaker": "A"}])
        actors = sm.Actors(actors_rows(lambda t: {"A": None if 6 <= t < 10 else (LEFT if t < 12.5 else CENTRE)}))
        trace = trace_rows(lambda t: ([(1, CENTRE)], 1) if t >= 12.5 else ([], None))
        result = sm.evaluate(truth, actors, trace, offset=0.0, kind="reentry", reentry_limit=5.0)
        self.assertAlmostEqual(result["reentry_latency"][0], 2.5, places=1)
        self.assertTrue(result["passed"])
        late = sm.evaluate(truth, actors, trace, offset=0.0, kind="reentry", reentry_limit=2.0)
        self.assertFalse(late["passed"])

    def test_walking_into_a_still_camera_is_not_a_reentry(self):
        # full1 exit_reenter n1: A walked into the centre of a wide, still
        # camera; the tracker took A 9.8 s later.
        truth = sm.parse_truth([{"start": 0, "end": 6, "speaker": "A"},
                                {"start": 6, "end": 10, "speaker": "none"},
                                {"start": 10, "end": 20, "speaker": "A"}])
        actors = sm.Actors(actors_rows(lambda t: {"A": None if 6 <= t < 10 else CENTRE}))
        trace = trace_rows(lambda t: ([(1, CENTRE)], 1 if t < 6 or t >= 14 else None))
        result = sm.evaluate(truth, actors, trace, offset=0.0, kind="reentry", reentry_limit=2.0)
        self.assertAlmostEqual(result["events"][-1]["camera_latency"], 0.0, places=1)
        self.assertAlmostEqual(result["reentry_latency"][0], 4.0, places=1)
        self.assertFalse(result["passed"])

    def test_alignment_finds_the_video_delay(self):
        def scene(t):
            x = 100.0 + 20.0 * t
            return {"A": (x, 100.0, x + 60.0, 300.0)}

        actors = sm.Actors(actors_rows(scene))
        # The tracker sees the scene 0.4 s late and its clock starts 1.0 s after Unity's.
        trace = trace_rows(lambda t: ([(1, scene(max(0.0, t + 1.0 - 0.4))["A"])], 1), end=15.0)
        offset, overlap = sm.align_offset(trace, actors, estimate=1.0)
        self.assertAlmostEqual(offset, 0.6, delta=0.03)
        self.assertGreater(overlap, 0.9)


if __name__ == "__main__":
    unittest.main()


class UnitySeriesRunnerTest(unittest.TestCase):
    def test_scenario_run_passes_the_scenario_and_saves_ground_truth(self):
        import argparse
        import json
        import subprocess
        import tempfile
        from unittest import mock

        import unity_series

        class FakeApi:
            def __init__(self):
                self.calls = []

            def json(self, method, path):
                self.calls.append((method, path))
                if path == "/status":
                    return {"state": "idle"} if len(self.calls) == 1 else {"state": "running"}
                if path.startswith("/run/start"):
                    return {"run_id": "run_0001"}
                return {}

            def call(self, method, path, timeout=10):
                self.calls.append((method, path))
                return f"data:{path}".encode()

        class FakeTracker:
            returncode = None

            def __init__(self, command, **kwargs):
                FakeTracker.command = command

            def wait(self, timeout=None):
                if not hasattr(self, "interrupted"):
                    raise subprocess.TimeoutExpired("tracker", timeout)
                return 0

            def send_signal(self, signal):
                self.interrupted = True

        api = FakeApi()
        args = argparse.Namespace(duration=None, path="", tracker_seconds=0.01, camera_host="sim",
                                  pipeline="pc", pc_detector="gpu", pc_reid="repvgg", record_fps=30.0,
                                  scenario_durations={"handover": 50.0}, owner="codex", wait_seconds=60.0,
                                  require_hfov=None)
        with tempfile.TemporaryDirectory() as directory, \
                mock.patch.object(unity_series, "OUT", Path(directory)), \
                mock.patch.object(unity_series.subprocess, "Popen", FakeTracker):
            for name in ("raw", "traces", "logs", "unity_metrics"):
                (Path(directory) / name).mkdir()
            unity_series.one_run(api, "t_handover_c1_1", "--marker-min-pixels 14", args,
                                 scenario="handover", label="c1")
            saved = sorted(p.name for p in (Path(directory) / "unity_metrics").iterdir())
            meta = json.loads((Path(directory) / "unity_metrics" / "t_handover_c1_1_meta.json").read_text())

        start = next(path for method, path in api.calls if path.startswith("/run/start"))
        self.assertIn("scenario=handover", start)
        self.assertIn("duration=50", start)
        self.assertIn("owner=codex", start)
        self.assertEqual(meta["duration_sec"], 50.0)
        self.assertEqual(
            unity_series.scenario_durations({"details": [{"name": "a", "durationSec": 40}]}), {"a": 40.0})
        self.assertEqual(saved, ["t_handover_c1_1_actors.csv", "t_handover_c1_1_frames.csv",
                                 "t_handover_c1_1_meta.json", "t_handover_c1_1_summary.json",
                                 "t_handover_c1_1_truth.json"])
        self.assertEqual((meta["scenario"], meta["label"], meta["pipeline"]), ("handover", "c1", "pc"))
        command = " ".join(FakeTracker.command)
        self.assertIn("--speaker-reid-onnx repvgg_a0_person_reid_512.onnx", command)
        self.assertIn("--inference-device cuda", command)
        self.assertIn("--speaker-reid-device cuda", command)
        self.assertIn("--record-input-fps 30.0", command)
        self.assertTrue(command.endswith("--marker-min-pixels 14"))


class FilesTest(unittest.TestCase):
    def test_run_files_are_read_and_aligned_by_the_log_clock(self):
        import csv
        import datetime
        import json
        import tempfile
        from argparse import Namespace
        from unittest import mock

        run_start = datetime.datetime(2026, 10, 25, 1, 0, 0, tzinfo=datetime.timezone.utc)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ("unity_metrics", "traces", "logs"):
                (root / name).mkdir()
            metrics = root / "unity_metrics"
            (metrics / "s_cross_c1_1_meta.json").write_text(json.dumps({"scenario": "cross_front", "label": "c1"}))
            with (metrics / "s_cross_c1_1_frames.csv").open("w", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(["time", "utc"])
                for k in range(400):
                    stamp = run_start + datetime.timedelta(seconds=0.05 * k)
                    writer.writerow([f"{0.05 * k:.2f}", stamp.isoformat().replace("+00:00", "Z")])
            (metrics / "s_cross_c1_1_truth.json").write_text(json.dumps([{"start": 0, "end": 20, "speaker": "A"}]))

            def scene(t):
                return {"A": CENTRE, "C": (100.0 + 20 * t, 100.0, 160.0 + 20 * t, 300.0)}

            rows = actors_rows(scene)
            with (metrics / "s_cross_c1_1_actors.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            # Tracker clock starts 2 s into the run; it follows C from 10 to 12 s (Unity time).
            trace = trace_rows(lambda t: ([(1, CENTRE), (3, scene(t + 2.0)["C"])],
                                          3 if 8.0 <= t < 10.0 else 1), end=15.0)
            (root / "traces" / "s_cross_c1_1.jsonl").write_text("\n".join(json.dumps(r) for r in trace))
            local = (run_start + datetime.timedelta(seconds=2)).astimezone().replace(tzinfo=None)
            (root / "logs" / "s_cross_c1_1.log").write_text(
                f"{local:%Y-%m-%d %H:%M:%S},000 INFO Tracking status: fps=0.0 target=None\n")
            options = Namespace(criterion_overrides={}, center_tolerance=0.15, match_iou=0.3, hold_seconds=0.5,
                                handover_limit=3.0, reentry_limit=5.0)
            with mock.patch.object(sm, "ROOT", root):
                result = sm.load_and_evaluate("s_cross_c1_1", options)
        self.assertEqual(result["kind"], "no_wrong")
        self.assertAlmostEqual(result["offset_seconds"], 2.0, delta=0.05)
        self.assertAlmostEqual(result["wrong_seconds"], 2.0, delta=0.1)
        self.assertEqual(result["wrong_episodes"][0]["actor"], "C")
        self.assertFalse(result["passed"])
        table = sm.markdown([result])
        self.assertIn("| cross_front | c1 |", table)
        self.assertIn("실패", table)

    def test_start_is_retried_after_409(self):
        import argparse
        import io
        import urllib.error
        from unittest import mock

        import unity_series

        class BusyOnceApi:
            def __init__(self):
                self.starts = 0

            def json(self, method, path):
                if path == "/status":
                    return {"state": "idle"}
                self.starts += 1
                if self.starts == 1:
                    raise urllib.error.HTTPError(path, 409, "busy", {}, io.BytesIO(b""))
                return {"run_id": "run_0002"}

        api = BusyOnceApi()
        with mock.patch.object(unity_series.time, "sleep"):
            _, reply = unity_series.start_run(api, "x", {"label": "x"}, argparse.Namespace(wait_seconds=60.0))
        self.assertEqual(reply["run_id"], "run_0002")
        self.assertEqual(api.starts, 2)

    def test_hfov_check_and_start_conditions(self):
        import unity_series

        status = {"preset": "ptz_default", "ptz": {"pan": 0, "tilt": 7.5, "hfov": 20.0, "zoom_pos": 5543},
                  "ptz_position": {"x": 0, "y": 2.45, "z": -5.6}}
        unity_series.check_hfov(status, 20.0, "x")
        with self.assertRaises(RuntimeError):
            unity_series.check_hfov({"ptz": {"hfov": 30}}, 20.0, "x")
        self.assertEqual(unity_series.start_conditions(status)["preset"], "ptz_default")
        self.assertEqual(unity_series.start_conditions(status)["hfov"], 20.0)
