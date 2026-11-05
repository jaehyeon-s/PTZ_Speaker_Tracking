import json
import unittest
from argparse import Namespace
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import parse_qs

from marker_tracker.app import (
    _camera_speed_limit_for_decision,
    ByteTrackDiagnostics,
    FramingPolicy,
    HeadAnchor,
    OcclusionAnchor,
    _to_detections,
    IdentityTraceRecorder,
    InputVideoRecorder,
    LoggingCameraAdapter,
    PerformanceDiagnostics,
    PredictIouPersonSource,
    ScoreDiagnostics,
    SimpleIouTracker,
    TrackingStatusDiagnostics,
    TrackContinuityDiagnostics,
    YoloPersonSource,
    _camera_target_for_decision,
    _create_camera_adapter,
    _apply_runtime_profile,
    _read_exported_imgsz,
    _log_person_tracking_configuration,
    _normalize_reid_feature_vectors,
    _recovery_status,
    _resolve_tracker_config,
    _resolve_inference_confidence,
    _tracker_config_summary,
)
from marker_tracker.camera import QonHttpCameraAdapter, ViscaOverIpCameraAdapter
from marker_tracker.core import BoundingBox, CandidateScore, Detection, HandoffDecision


class _Value:
    def __init__(self, value):
        self._value = value

    def item(self):
        return self._value


class _ListValue:
    def __init__(self, value):
        self._value = value

    def tolist(self):
        return self._value


class _FakeBox:
    def __init__(self, class_id, confidence, xyxy, track_id=None):
        self.cls = [_Value(class_id)]
        self.conf = [_Value(confidence)]
        self.xyxy = [_ListValue(xyxy)]
        self.id = None if track_id is None else _ListValue([track_id])


class _FakeFrame:
    shape = (100, 200, 3)

    def __getitem__(self, key):
        return key


class _RecordingWriter:
    def __init__(self):
        self.frames = []
        self.released = False

    def isOpened(self):
        return True

    def write(self, frame):
        self.frames.append(frame)

    def release(self):
        self.released = True


class _RecordingCv2:
    def __init__(self):
        self.writer = _RecordingWriter()
        self.writer_args = None

    @staticmethod
    def VideoWriter_fourcc(*codec):
        return codec

    def VideoWriter(self, *args):
        self.writer_args = args
        return self.writer


class _FakeSocket:
    def __init__(self):
        self.sent = []
        self.closed = False

    def settimeout(self, timeout):
        self.timeout = timeout

    def sendto(self, packet, address):
        self.sent.append((packet, address))

    def close(self):
        self.closed = True


class _FakeHttpResponse:
    status = 200

    def __init__(self, body="ok"):
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def read(self):
        return self.body.encode("utf-8")


class _FakeHttpOpener:
    def __init__(self):
        self.requests = []
        self.body = (
            'common.track="0"\n'
            'tracking.auto_zoom="0"\n'
            'tracking.auto_tilt="0"\n'
            'common.debug_mode="3"'
        )

    def __call__(self, request, timeout):
        self.requests.append((request, timeout))
        return _FakeHttpResponse(self.body)


class _FakeYoloModel:
    def __init__(self):
        self.names = {0: "person", 1: "bicycle"}
        self.track_kwargs = None
        self.predict_kwargs = None

    def track(self, frame, **kwargs):
        self.track_kwargs = kwargs
        result = Namespace(
            names={0: "person"},
            boxes=[_FakeBox(0, 0.9, [10, 20, 30, 60], track_id=5)],
        )
        return [result]

    def predict(self, frame, **kwargs):
        self.predict_kwargs = kwargs
        result = Namespace(
            names={0: "person"},
            boxes=[_FakeBox(0, 0.9, [10, 20, 30, 60])],
        )
        return [result]


class TargetLowConfidenceTest(unittest.TestCase):
    def _detections(self, hint, low=0.25):
        boxes = [
            _FakeBox(0, 0.30, [100, 50, 140, 110], track_id=7),
            _FakeBox(0, 0.30, [400, 50, 440, 110], track_id=9),
            _FakeBox(0, 0.20, [100, 50, 140, 110], track_id=7),
            _FakeBox(0, 0.30, [104, 54, 140, 100], track_id=12),
        ]
        return _to_detections({0: "person"}, boxes, 0.35, target_low_confidence=low, target_hint=hint)

    def test_weak_detections_are_dropped_by_default(self):
        self.assertEqual(self._detections((7, BoundingBox(100, 50, 140, 200)), low=None), [])
        self.assertEqual(self._detections(None), [])

    def test_weak_detection_of_target_track_or_place_is_kept(self):
        kept = self._detections((7, BoundingBox(100, 50, 140, 200)))
        # Track 7 at 0.30 and the new track 12 in the same place; not the far
        # person, and nothing below the low limit.
        self.assertEqual([(d.track_id, d.confidence) for d in kept], [(7, 0.30), (12, 0.30)])


class OcclusionAnchorTest(unittest.TestCase):
    def test_short_box_keeps_remembered_height_from_same_top(self):
        anchor = OcclusionAnchor(height_ratio=0.6, hold_seconds=5.0)
        full = BoundingBox(100, 50, 160, 250)
        for step in range(5):
            self.assertEqual(anchor.adjust(full, 0.85, step * 0.1), full)
        head = BoundingBox(110, 52, 150, 110)
        adjusted = anchor.adjust(head, 0.40, 0.6)
        self.assertEqual((adjusted.x1, adjusted.y1, adjusted.x2), (110, 52, 150))
        self.assertAlmostEqual(adjusted.y2 - adjusted.y1, 200, delta=1)
        # A short box that is detected just as confidently is a real change.
        confident = OcclusionAnchor()
        confident.adjust(full, 0.85, 0.0)
        self.assertEqual(confident.adjust(head, 0.85, 0.1), head)

    def test_occlusion_longer_than_hold_becomes_the_new_reference(self):
        anchor = OcclusionAnchor(height_ratio=0.6, hold_seconds=1.0)
        anchor.adjust(BoundingBox(100, 50, 160, 250), 0.85, 0.0)
        head = BoundingBox(110, 52, 150, 110)
        self.assertNotEqual(anchor.adjust(head, 0.4, 0.5), head)
        self.assertEqual(anchor.adjust(head, 0.4, 2.0), head)

    def test_framing_point_stays_on_the_body_when_anchored(self):
        anchor = OcclusionAnchor()
        full = Detection("person", 0.85, BoundingBox(100, 50, 160, 250), track_id=7)
        head = Detection("person", 0.40, BoundingBox(110, 52, 150, 110), track_id=7)
        framing = FramingPolicy("upper-body")
        shape = (360, 640, 3)
        def target(detection, now, use_anchor):
            decision = HandoffDecision("tracking", detection.box, None, 7)
            return _camera_target_for_decision(
                decision, framing, shape, anchor if use_anchor else None, [detection], now
            ).center[1]
        body_point = target(full, 0.0, True)
        self.assertAlmostEqual(target(head, 0.1, True), body_point, delta=4)
        self.assertLess(target(head, 0.1, False), body_point - 40)


class HeadAnchorTest(unittest.TestCase):
    def test_framing_point_ignores_a_jumping_box_bottom(self):
        framing = FramingPolicy("upper-body")
        shape = (360, 640, 3)
        def points(anchor):
            out = []
            for step in range(20):
                bottom = 300 if step % 2 == 0 else 250  # legs hidden on every other frame
                decision = HandoffDecision("tracking", BoundingBox(300, 100, 340, bottom), None, 7)
                out.append(_camera_target_for_decision(decision, framing, shape, None, [], step * 0.07, anchor).center[1])
            return out
        plain = points(None)
        anchored = points(HeadAnchor(0.5))
        self.assertGreater(max(plain[4:]) - min(plain[4:]), 15)
        self.assertLess(max(anchored[4:]) - min(anchored[4:]), 6)

    def test_top_edge_is_smoothed_and_reset_on_release(self):
        anchor = HeadAnchor(0.5)
        anchor.adjust(BoundingBox(300, 100, 340, 300), 360, 0.0)
        moved = anchor.adjust(BoundingBox(300, 140, 340, 340), 360, 0.1)
        self.assertAlmostEqual(moved.y1, 108, delta=1)
        anchor.reset()
        self.assertEqual(anchor.adjust(BoundingBox(300, 140, 340, 340), 360, 0.2).y1, 140)


class YoloPersonSourceTest(unittest.TestCase):
    def test_passes_lower_tracker_input_confidence_without_lowering_person_filter(self):
        model = _FakeYoloModel()
        source = YoloPersonSource(
            Namespace(confidence=None, person_confidence=0.35, imgsz=416)
        )

        _, detections = source.update(
            frame=object(),
            model=model,
            inference_confidence=0.1,
            bytetrack_config="bytetrack-pi.yaml",
        )

        self.assertEqual(model.track_kwargs["conf"], 0.1)
        self.assertEqual(model.track_kwargs["classes"], [0])
        self.assertEqual(detections[0].confidence, 0.9)

    def test_passes_configured_imgsz_to_model_track(self):
        model = _FakeYoloModel()
        source = YoloPersonSource(
            Namespace(confidence=None, person_confidence=0.35, imgsz=576)
        )

        result, detections = source.update(
            frame=object(),
            model=model,
            inference_confidence=0.35,
            bytetrack_config="bytetrack.yaml",
        )

        self.assertEqual(model.track_kwargs["imgsz"], 576)
        self.assertEqual(result.names, {0: "person"})
        self.assertEqual(detections[0].track_id, 5)

    def test_passes_explicit_inference_device_to_nested_tracker_models(self):
        model = _FakeYoloModel()
        source = YoloPersonSource(
            Namespace(
                confidence=None,
                person_confidence=0.35,
                imgsz=416,
                inference_device="cpu",
            )
        )

        source.update(
            frame=object(),
            model=model,
            inference_confidence=0.1,
            bytetrack_config="botsort-pi.yaml",
        )

        self.assertEqual(model.track_kwargs["device"], "cpu")

    def test_omits_imgsz_when_not_configured_for_legacy_track_path(self):
        model = _FakeYoloModel()
        source = YoloPersonSource(
            Namespace(confidence=None, person_confidence=0.35, imgsz=None)
        )

        source.update(
            frame=object(),
            model=model,
            inference_confidence=0.35,
            bytetrack_config="bytetrack.yaml",
        )

        self.assertNotIn("imgsz", model.track_kwargs)

    def test_simple_iou_source_uses_model_predict_and_assigns_track_id(self):
        model = _FakeYoloModel()
        source = PredictIouPersonSource(
            Namespace(
                confidence=None,
                person_confidence=0.35,
                imgsz=640,
                simple_iou_threshold=0.3,
                simple_iou_max_lost_frames=15,
            )
        )

        result, detections = source.update(
            frame=object(),
            model=model,
            inference_confidence=0.35,
            bytetrack_config="bytetrack.yaml",
        )

        self.assertEqual(model.predict_kwargs["imgsz"], 640)
        self.assertEqual(result.names, {0: "person"})
        self.assertEqual(detections[0].track_id, 1)

    def test_simple_iou_source_omits_imgsz_when_not_configured(self):
        model = _FakeYoloModel()
        source = PredictIouPersonSource(
            Namespace(
                confidence=None,
                person_confidence=0.35,
                imgsz=None,
                simple_iou_threshold=0.3,
                simple_iou_max_lost_frames=15,
            )
        )

        source.update(
            frame=object(),
            model=model,
            inference_confidence=0.35,
            bytetrack_config="bytetrack.yaml",
        )

        self.assertNotIn("imgsz", model.predict_kwargs)


class SimpleIouTrackerTest(unittest.TestCase):
    def test_keeps_track_id_for_overlapping_person(self):
        tracker = SimpleIouTracker(match_threshold=0.3)
        first = tracker.update(
            [Detection("person", 0.9, BoundingBox(10, 20, 30, 60))]
        )
        second = tracker.update(
            [Detection("person", 0.8, BoundingBox(12, 22, 32, 62))]
        )

        self.assertEqual(first[0].track_id, 1)
        self.assertEqual(second[0].track_id, 1)

    def test_creates_new_track_for_non_overlapping_person(self):
        tracker = SimpleIouTracker(match_threshold=0.3)
        tracker.update([Detection("person", 0.9, BoundingBox(10, 20, 30, 60))])
        second = tracker.update(
            [Detection("person", 0.8, BoundingBox(100, 120, 130, 160))]
        )

        self.assertEqual(second[0].track_id, 2)


class ByteTrackDiagnosticsTest(unittest.TestCase):
    def test_logs_recommended_track_buffer_from_fps(self):
        now = iter([10.0, 13.0])
        diagnostics = ByteTrackDiagnostics(
            expected_buffer_seconds=3.0,
            log_interval_seconds=2.0,
            now_fn=lambda: next(now),
        )
        detections = [Detection("marker", 0.8, BoundingBox(20, 30, 50, 70), track_id=4)]
        person_detection = Detection("person", 0.9, BoundingBox(0, 0, 100, 200), track_id=7)

        diagnostics.update([person_detection, *detections], fps=25.0, target_track_id=7)
        with self.assertLogs(level="INFO") as logs:
            diagnostics.update(
                [person_detection, *detections],
                fps=25.0,
                target_track_id=7,
            )

        self.assertIn("recommended_track_buffer=75", "\n".join(logs.output))


class PerformanceDiagnosticsTest(unittest.TestCase):
    def test_logs_average_and_p95_stage_timings(self):
        now = iter([10.0, 16.0])
        diagnostics = PerformanceDiagnostics(
            interval_seconds=5.0,
            now_fn=lambda: next(now),
        )
        diagnostics.update(0.010, 0.002, 0.001, 0.015)

        with self.assertLogs(level="INFO") as logs:
            diagnostics.update(0.020, 0.004, 0.003, 0.030)

        output = "\n".join(logs.output)
        self.assertIn("Performance diagnostics", output)
        self.assertIn("samples=2", output)
        self.assertIn("person_source_avg_ms=15.00", output)
        self.assertIn("person_source_p95_ms=20.00", output)
        self.assertIn("pipeline_avg_ms=22.50", output)

    def test_logs_optional_sub_stage_timings(self):
        now = iter([10.0, 16.0])
        diagnostics = PerformanceDiagnostics(interval_seconds=5.0, now_fn=lambda: next(now))
        extra = {"person_detect": 0.008, "person_track": 0.002}
        diagnostics.update(0.010, 0.002, 0.001, 0.015, extra_seconds=extra)

        with self.assertLogs(level="INFO") as logs:
            diagnostics.update(0.010, 0.002, 0.001, 0.015, extra_seconds=extra)

        output = "\n".join(logs.output)
        self.assertIn("person_detect_avg_ms=8.00", output)
        self.assertIn("person_track_avg_ms=2.00", output)


class TrackerConfigTest(unittest.TestCase):
    def test_resolves_bytetrack_project_config_by_default(self):
        args = Namespace(
            tracker_config=None,
            tracker_preset="bytetrack",
            bytetrack_config="bytetrack.yaml",
        )

        self.assertEqual(_resolve_tracker_config(args), "bytetrack.yaml")

    def test_resolves_botsort_preset(self):
        args = Namespace(
            tracker_config=None,
            tracker_preset="botsort",
            bytetrack_config="bytetrack.yaml",
        )

        self.assertEqual(_resolve_tracker_config(args), "botsort.yaml")

    def test_explicit_tracker_config_wins(self):
        args = Namespace(
            tracker_config="custom.yaml",
            tracker_preset="botsort",
            bytetrack_config="bytetrack.yaml",
        )

        self.assertEqual(_resolve_tracker_config(args), "custom.yaml")

    def test_botsort_pi_summary_reports_gmc_and_disabled_reid(self):
        summary = _tracker_config_summary("botsort-pi.yaml")

        self.assertIn("tracker_type=botsort", summary)
        self.assertIn("gmc_method=sparseOptFlow", summary)
        self.assertIn("with_reid=False", summary)

    def test_normalizes_batched_onnx_reid_features_for_association(self):
        import numpy as np

        normalized = _normalize_reid_feature_vectors(
            [np.zeros((1, 512), dtype=np.float32)]
        )

        self.assertEqual(len(normalized), 1)
        self.assertEqual(normalized[0].shape, (512,))


class RuntimeProfileTest(unittest.TestCase):
    def test_pi_qon_http_profile_applies_field_defaults(self):
        args = Namespace(
            runtime_profile="pi-qon-http",
            model="yolo26n.pt",
            imgsz=640,
            bytetrack_config="bytetrack.yaml",
            aruco_marker_id=None,
            aruco_detect_flip="none",
            confirmation_seconds=3.0,
            release_grace_seconds=3.0,
            marker_dropout_seconds=0.5,
            switch_score_margin=12.0,
            switch_cooldown_seconds=1.0,
            area_growth_penalty_threshold=1.8,
            tracking_status_diagnostics=False,
            framing_mode="center",
            track_buffer_seconds=3.0,
            camera_control="log",
            qon_min_speed=2,
            qon_max_speed=10,
            qon_deadzone=0.12,
        )

        profiled = _apply_runtime_profile(args, provided_dests=set())

        self.assertEqual(profiled.model, "yolo26n_ncnn_model")
        self.assertEqual(profiled.imgsz, 416)
        self.assertEqual(profiled.inference_device, "cpu")
        self.assertEqual(profiled.bytetrack_config, "bytetrack-pi.yaml")
        self.assertEqual(profiled.aruco_marker_id, 0)
        self.assertEqual(profiled.aruco_detect_flip, "unmirrored")
        self.assertEqual(profiled.confirmation_seconds, 2.0)
        self.assertEqual(profiled.marker_dropout_seconds, 1.5)
        self.assertEqual(profiled.speaker_resolution_timeout_seconds, 10.0)
        self.assertEqual(profiled.provisional_speed_limit, 4)
        self.assertEqual(profiled.qon_full_speed_error, 0.5)
        self.assertTrue(profiled.qon_uniform_error_scale)
        self.assertEqual(profiled.qon_axis_switch_ratio, 1.5)
        self.assertEqual(profiled.qon_response_delay_seconds, 0.5)
        self.assertEqual(profiled.tracker_input_confidence, 0.1)
        self.assertEqual(profiled.recovery_confirmation_seconds, 0.12)
        self.assertEqual(profiled.recovery_min_iou, 0.2)
        self.assertEqual(profiled.recovery_max_center_distance, 0.75)
        self.assertEqual(profiled.recovery_score_margin, 0.25)
        self.assertEqual(profiled.speaker_resolution_min_frames, 8)
        self.assertEqual(profiled.speaker_resolution_min_seconds, 0.50)
        self.assertEqual(profiled.speaker_resolution_min_margin, 0.06)
        self.assertEqual(profiled.speaker_resolution_winner_ratio, 0.85)
        self.assertEqual(profiled.speaker_resolution_search_after_seconds, 1.2)
        self.assertEqual(profiled.speaker_resolution_search_interval_frames, 2)
        self.assertEqual(profiled.speaker_single_resolution_min_frames, 5)
        self.assertEqual(profiled.speaker_single_resolution_min_seconds, 0.35)
        self.assertEqual(profiled.speaker_single_resolution_min_similarity, 0.50)
        self.assertEqual(profiled.speaker_coast_max_seconds, 1.2)
        self.assertTrue(profiled.tracking_status_diagnostics)
        self.assertTrue(profiled.track_continuity_diagnostics)
        self.assertTrue(profiled.performance_diagnostics)
        self.assertEqual(profiled.framing_mode, "upper-body")
        self.assertEqual(profiled.camera_control, "qon-http")
        self.assertEqual(profiled.qon_deadzone, 0.10)
        self.assertEqual(profiled.qon_stop_deadzone, 0.07)
        self.assertEqual(profiled.qon_stop_debounce_seconds, 0.08)
        self.assertEqual(profiled.qon_lost_target_hold_seconds, 0.35)
        self.assertEqual(profiled.qon_command_min_interval_seconds, 0.08)

    def test_runtime_profile_does_not_replace_explicit_options(self):
        args = Namespace(
            runtime_profile="pi-qon-http",
            model="custom.pt",
            imgsz=512,
            bytetrack_config="bytetrack.yaml",
            aruco_marker_id=None,
            aruco_detect_flip="none",
            confirmation_seconds=1.5,
            release_grace_seconds=3.0,
            marker_dropout_seconds=0.5,
            switch_score_margin=12.0,
            switch_cooldown_seconds=1.0,
            area_growth_penalty_threshold=1.8,
            speaker_coast_max_seconds=0.6,
            tracking_status_diagnostics=False,
            framing_mode="center",
            track_buffer_seconds=3.0,
            camera_control="log",
            qon_min_speed=2,
            qon_max_speed=10,
            qon_deadzone=0.12,
        )

        profiled = _apply_runtime_profile(
            args,
            provided_dests={
                "model",
                "confirmation_seconds",
                "speaker_coast_max_seconds",
            },
        )

        self.assertEqual(profiled.model, "custom.pt")
        self.assertEqual(profiled.confirmation_seconds, 1.5)
        self.assertEqual(profiled.speaker_coast_max_seconds, 0.6)
        self.assertEqual(profiled.imgsz, 512)
        self.assertEqual(profiled.camera_control, "qon-http")

    def test_runtime_profile_does_not_force_imgsz_when_model_is_explicit(self):
        args = Namespace(
            runtime_profile="pi-qon-http",
            model="yolo26n.pt",
            imgsz=None,
            bytetrack_config="bytetrack.yaml",
            aruco_marker_id=None,
            aruco_detect_flip="none",
            confirmation_seconds=3.0,
            release_grace_seconds=3.0,
            marker_dropout_seconds=0.5,
            switch_score_margin=12.0,
            switch_cooldown_seconds=1.0,
            area_growth_penalty_threshold=1.8,
            tracking_status_diagnostics=False,
            framing_mode="center",
            track_buffer_seconds=3.0,
            camera_control="log",
            qon_min_speed=2,
            qon_max_speed=10,
            qon_deadzone=0.12,
        )

        profiled = _apply_runtime_profile(args, provided_dests={"model"})

        self.assertEqual(profiled.model, "yolo26n.pt")
        self.assertIsNone(profiled.imgsz)
        self.assertEqual(profiled.camera_control, "qon-http")


class ModelMetadataTest(unittest.TestCase):
    def test_reads_exported_imgsz_from_ncnn_metadata(self):
        with TemporaryDirectory() as directory:
            metadata_path = f"{directory}/metadata.yaml"
            with open(metadata_path, "w", encoding="utf-8") as handle:
                handle.write("imgsz:\n- 640\n- 640\n")

            self.assertEqual(_read_exported_imgsz(directory), (640, 640))


class InferenceConfidenceTest(unittest.TestCase):
    def test_tracker_input_confidence_is_independent_from_person_confidence(self):
        args = Namespace(
            tracker_input_confidence=0.1,
            confidence=None,
            person_confidence=0.35,
        )

        self.assertEqual(_resolve_inference_confidence(args), 0.1)

    def test_legacy_confidence_is_used_without_tracker_override(self):
        args = Namespace(
            tracker_input_confidence=None,
            confidence=0.2,
            person_confidence=0.35,
        )

        self.assertEqual(_resolve_inference_confidence(args), 0.2)


class ScoreDiagnosticsTest(unittest.TestCase):
    def test_logs_candidate_scores(self):
        diagnostics = ScoreDiagnostics(interval_seconds=1.0, now_fn=lambda: 10.0)
        decision = HandoffDecision(
            "tracking",
            BoundingBox(0, 0, 100, 200),
            BoundingBox(20, 40, 40, 80),
            target_track_id=7,
            candidate_scores=(
                CandidateScore(
                    track_id=7,
                    score=42.5,
                    box=BoundingBox(0, 0, 100, 200),
                    is_active=True,
                    area_growth_ratio=1.2,
                ),
            ),
        )

        with self.assertLogs(level="INFO") as logs:
            diagnostics.update(decision)

        self.assertIn("Candidate scores", "\n".join(logs.output))
        self.assertIn("track=7", "\n".join(logs.output))


class TrackingStatusDiagnosticsTest(unittest.TestCase):
    def test_logs_state_reason_ptz_and_recovery(self):
        diagnostics = TrackingStatusDiagnostics(interval_seconds=1.0, now_fn=lambda: 10.0)
        decision = HandoffDecision(
            "switching",
            BoundingBox(0, 0, 100, 200),
            BoundingBox(20, 40, 40, 80),
            target_track_id=7,
            reason="switch_candidate_confirming",
            candidate_scores=(
                CandidateScore(
                    track_id=8,
                    score=51.2,
                    box=BoundingBox(100, 0, 200, 200),
                ),
            ),
        )

        with self.assertLogs(level="INFO") as logs:
            diagnostics.update(
                decision,
                [
                    Detection("person", 0.9, BoundingBox(0, 0, 100, 200), track_id=7),
                    Detection("marker", 1.0, BoundingBox(20, 40, 40, 80)),
                ],
                fps=24.5,
                ptz_status="right:5",
                framing_mode="board-right",
            )

        output = "\n".join(logs.output)
        self.assertIn("Tracking status", output)
        self.assertIn("reason=switch_candidate_confirming", output)
        self.assertIn("marker_score=51.2", output)
        self.assertIn("tracks=7:0.90", output)
        self.assertIn("ptz=right:5", output)
        self.assertIn("recovery=handoff_confirmation", output)

    def test_recovery_status_reports_lost_target_grace(self):
        decision = HandoffDecision(
            "grace",
            BoundingBox(0, 0, 100, 200),
            None,
            target_track_id=7,
            reason="target_lost_grace",
        )

        self.assertEqual(
            _recovery_status(decision, marker_count=0, people_count=0),
            "holding_last_target",
        )


class TrackContinuityDiagnosticsTest(unittest.TestCase):
    def test_logs_replacement_candidate_when_target_id_disappears(self):
        diagnostics = TrackContinuityDiagnostics()
        target = Detection("person", 0.9, BoundingBox(0, 0, 100, 200), track_id=7)
        replacement = Detection(
            "person", 0.8, BoundingBox(10, 0, 110, 200), track_id=12
        )
        tracking = HandoffDecision(
            "tracking", target.box, None, target_track_id=7, reason="active_track_visible"
        )
        grace = HandoffDecision(
            "grace", target.box, None, target_track_id=7, reason="target_lost_grace"
        )

        with self.assertLogs(level="INFO") as logs:
            diagnostics.update(tracking, [target], now=10.0)
            diagnostics.update(grace, [replacement], now=10.5)
            moved_replacement = Detection(
                "person", 0.75, BoundingBox(12, 0, 112, 200), track_id=12
            )
            diagnostics.update(grace, [moved_replacement], now=10.6)

        output = "\n".join(logs.output)
        self.assertIn("event=target_missing", output)
        self.assertIn("target_id=7", output)
        self.assertIn("track=12", output)
        self.assertIn("iou=0.818", output)
        self.assertNotIn("event=missing_candidates_changed", output)

    def test_logs_same_id_reappearance_and_release_duration(self):
        diagnostics = TrackContinuityDiagnostics()
        target = Detection("person", 0.9, BoundingBox(0, 0, 100, 200), track_id=7)
        tracking = HandoffDecision("tracking", target.box, None, target_track_id=7)
        grace = HandoffDecision("grace", target.box, None, target_track_id=7)
        released = HandoffDecision(
            "default",
            target.box,
            None,
            target_track_id=7,
            reason="target_released_after_grace",
            emit_release=True,
        )

        with self.assertLogs(level="INFO") as logs:
            diagnostics.update(tracking, [target], now=10.0)
            diagnostics.update(grace, [], now=10.5)
            diagnostics.update(tracking, [target], now=11.0)
            diagnostics.update(grace, [], now=12.0)
            diagnostics.update(released, [], now=15.0)

        output = "\n".join(logs.output)
        self.assertIn("event=target_reappeared", output)
        self.assertIn("missing_seconds=0.500", output)
        self.assertIn("event=target_released", output)
        self.assertIn("missing_seconds=3.000", output)

    def test_provisional_recovery_logs_metrics_against_previous_box(self):
        diagnostics = TrackContinuityDiagnostics()
        target = Detection("person", 0.9, BoundingBox(0, 0, 100, 200), track_id=7)
        replacement = Detection(
            "person", 0.8, BoundingBox(10, 0, 60, 100), track_id=12
        )
        tracking = HandoffDecision("tracking", target.box, None, target_track_id=7)
        recovering = HandoffDecision(
            "recovering",
            replacement.box,
            None,
            target_track_id=7,
            provisional_target=True,
        )

        with self.assertLogs(level="INFO") as logs:
            diagnostics.update(tracking, [target], now=10.0)
            diagnostics.update(recovering, [replacement], now=10.1)

        output = "\n".join(logs.output)
        self.assertIn("area_ratio=0.250", output)
        self.assertIn("width_ratio=0.500", output)
        self.assertNotIn("area_ratio=1.000", output)


class PersonTrackingConfigurationDiagnosticsTest(unittest.TestCase):
    def test_logs_simple_iou_backend_explicitly(self):
        args = Namespace(
            person_source="yolo",
            person_tracker="simple-iou",
            model="model.ncnn",
            imgsz=416,
            confidence=None,
            person_confidence=0.35,
            simple_iou_threshold=0.3,
            simple_iou_max_lost_frames=15,
        )

        with self.assertLogs(level="INFO") as logs:
            _log_person_tracking_configuration(args, "bytetrack-pi.yaml", 0.35)

        output = "\n".join(logs.output)
        self.assertIn("backend=simple-iou", output)
        self.assertIn("tracker_config=ignored", output)
        self.assertIn("max_lost_frames=15", output)

    def test_logs_fixture_source_without_claiming_a_tracker_backend(self):
        args = Namespace(
            person_source="fixture",
            person_tracker="ultralytics",
            model="unused.pt",
            imgsz=None,
            confidence=None,
            person_confidence=0.35,
        )

        with self.assertLogs(level="INFO") as logs:
            _log_person_tracking_configuration(args, "bytetrack.yaml", 0.35)

        output = "\n".join(logs.output)
        self.assertIn("person_source=fixture", output)
        self.assertIn("backend=fixture", output)
        self.assertIn("tracker_config=ignored", output)


class FramingPolicyTest(unittest.TestCase):
    def test_identity_resolution_tracks_visible_active_target(self):
        box = BoundingBox(100, 100, 300, 500)
        decision = HandoffDecision(
            "identity_resolving",
            box,
            None,
            target_track_id=2,
            reason="identity_evidence_collecting",
        )

        self.assertEqual(
            _camera_target_for_decision(
                decision,
                FramingPolicy("center"),
                (720, 1280, 3),
            ),
            box,
        )

    def test_identity_resolution_without_an_estimate_gives_no_camera_target(self):
        decision = HandoffDecision(
            "identity_resolving",
            None,
            None,
            target_track_id=7,
            reason="identity_searching",
        )

        target = _camera_target_for_decision(
            decision,
            FramingPolicy("upper-body"),
            (360, 640, 3),
        )

        self.assertIsNone(target)

    def test_provisional_identity_target_uses_the_configured_speed_limit(self):
        provisional = HandoffDecision(
            "identity_resolving",
            BoundingBox(0, 0, 100, 200),
            None,
            target_track_id=7,
            provisional_target=True,
        )
        confirmed = HandoffDecision(
            "tracking",
            BoundingBox(0, 0, 100, 200),
            None,
            target_track_id=7,
        )

        self.assertEqual(
            _camera_speed_limit_for_decision(provisional, Namespace(provisional_speed_limit=4)),
            4,
        )
        self.assertEqual(_camera_speed_limit_for_decision(provisional, Namespace()), 2)
        self.assertIsNone(
            _camera_speed_limit_for_decision(confirmed, Namespace(provisional_speed_limit=4))
        )

    def test_identity_resolution_can_soft_coast_on_a_provisional_path(self):
        box = BoundingBox(100, 100, 300, 500)
        decision = HandoffDecision(
            "identity_resolving",
            box,
            None,
            target_track_id=2,
            reason="identity_evidence_collecting",
            provisional_target=True,
        )

        self.assertEqual(
            _camera_target_for_decision(
                decision,
                FramingPolicy("center"),
                (720, 1280, 3),
            ),
            box,
        )

    def test_grace_and_unvalidated_recovery_stop_using_stale_camera_target(self):
        box = BoundingBox(100, 100, 300, 500)
        framing = FramingPolicy("center")

        for state in ("grace", "recovering"):
            decision = HandoffDecision(state, box, None, target_track_id=7)
            self.assertIsNone(
                _camera_target_for_decision(decision, framing, (720, 1280, 3))
            )

    def test_recovery_can_follow_a_validated_provisional_target(self):
        box = BoundingBox(100, 100, 300, 500)
        decision = HandoffDecision(
            "recovering",
            box,
            None,
            target_track_id=7,
            provisional_target=True,
        )

        self.assertEqual(
            _camera_target_for_decision(
                decision, FramingPolicy("center"), (720, 1280, 3)
            ),
            box,
        )

    def test_tracking_uses_the_current_camera_target(self):
        box = BoundingBox(100, 100, 300, 500)
        decision = HandoffDecision("tracking", box, None, target_track_id=7)

        self.assertEqual(
            _camera_target_for_decision(
                decision, FramingPolicy("center"), (720, 1280, 3)
            ),
            box,
        )

    def test_upper_body_moves_camera_target_above_box_center(self):
        target = FramingPolicy("upper-body").camera_target(
            BoundingBox(100, 100, 300, 500),
            (720, 1280, 3),
        )

        self.assertEqual(target.center, (200.0, 240.0))

    def test_board_right_places_speaker_left_of_center_to_leave_right_space(self):
        target = FramingPolicy("board-right").camera_target(
            BoundingBox(100, 100, 300, 500),
            (720, 1280, 3),
        )

        self.assertEqual(target.center, (164.0, 300.0))


class AppearanceEncoderTest(unittest.TestCase):
    def test_crops_are_encoded_one_per_call(self):
        try:
            import numpy as np
            import torch  # noqa: F401
            from ultralytics.utils.plotting import save_one_box  # noqa: F401
        except ImportError:
            self.skipTest("Ultralytics is not installed")
        from marker_tracker.app import UltralyticsAppearanceEncoder

        batch_sizes = []

        class _Model:
            @staticmethod
            def predictor(crops):
                batch_sizes.append(len(crops))
                return [np.full((1, 4), float(crop.shape[1])) for crop in crops]

        encoder = UltralyticsAppearanceEncoder.__new__(UltralyticsAppearanceEncoder)
        encoder._model = _Model()
        frame = np.zeros((360, 640, 3), dtype=np.uint8)
        people = [
            Detection("person", 0.9, BoundingBox(10, 10, 110, 310), track_id=1),
            Detection("person", 0.9, BoundingBox(300, 50, 360, 200), track_id=2),
            Detection("person", 0.9, BoundingBox(400, 20, 600, 350), track_id=3),
        ]

        features = encoder.encode(frame, people)

        # A mixed-shape batch is padded to the full square model input, which
        # is several times slower than three minimal-rectangle calls.
        self.assertEqual(batch_sizes, [1, 1, 1])
        self.assertEqual([feature.shape for feature in features], [(4,)] * 3)
        self.assertLess(features[1][0], features[0][0])
        self.assertLess(features[0][0], features[2][0])


class RecordingDiagnosticsTest(unittest.TestCase):
    def test_input_video_recorder_writes_pre_overlay_frames(self):
        cv2 = _RecordingCv2()
        with TemporaryDirectory() as directory:
            path = Path(directory) / "raw" / "crossing.mp4"
            recorder = InputVideoRecorder(cv2, str(path), 15.0)
            frame = _FakeFrame()

            recorder.write(frame)
            recorder.close()

        self.assertEqual(cv2.writer_args[0], str(path))
        self.assertEqual(cv2.writer_args[2], 15.0)
        self.assertEqual(cv2.writer_args[3], (200, 100))
        self.assertEqual(cv2.writer.frames, [frame])
        self.assertTrue(cv2.writer.released)

    def test_identity_trace_records_detections_and_decision(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "trace" / "crossing.jsonl"
            recorder = IdentityTraceRecorder(path)
            detection = Detection(
                "person",
                0.8,
                BoundingBox(1, 2, 3, 4),
                track_id=7,
                appearance_similarity=0.42,
            )
            decision = HandoffDecision(
                "identity_resolving",
                detection.box,
                None,
                target_track_id=2,
                reason="identity_evidence_collecting",
                speaker_id=1,
            )

            recorder.write(12, 3.5, [detection], decision)
            recorder.close()
            payload = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(payload["frame"], 12)
        self.assertEqual(payload["detections"][0]["track_id"], 7)
        self.assertEqual(payload["detections"][0]["appearance_similarity"], 0.42)
        self.assertEqual(payload["decision"]["state"], "identity_resolving")


class CameraAdapterTest(unittest.TestCase):
    def _visca(self):
        now = iter([1.0, 2.0, 3.0])
        adapter = ViscaOverIpCameraAdapter(
            "192.0.2.10",
            pan_speed=8,
            tilt_speed=6,
            min_interval_seconds=0,
            now_fn=lambda: next(now),
            socket_factory=lambda family, kind: _FakeSocket(),
        )
        return adapter, adapter._socket

    def test_log_camera_adapter_is_default(self):
        args = Namespace(camera_control="log")

        self.assertIsInstance(_create_camera_adapter(args), LoggingCameraAdapter)

    def test_visca_camera_requires_host(self):
        args = Namespace(camera_control="visca", visca_host="")

        with self.assertRaises(SystemExit):
            _create_camera_adapter(args)

    def test_qon_http_camera_requires_url_or_rtsp_host(self):
        args = Namespace(camera_control="qon-http", qon_camera_url="", rtsp_url="")

        with self.assertRaises(SystemExit):
            _create_camera_adapter(args)

    def test_qon_http_camera_url_defaults_to_rtsp_host(self):
        args = Namespace(
            camera_control="qon-http",
            qon_camera_url="",
            rtsp_url="rtsp://user:pass@192.0.2.20:554/stream2",
            qon_username=None,
            qon_password=None,
            qon_password_env=None,
            qon_auth_mode="basic",
            qon_min_speed=2,
            qon_max_speed=10,
            qon_deadzone=0.12,
            qon_command_ttl_seconds=0.35,
            qon_timeout_seconds=1.0,
            no_qon_configure_supervisor_actuator=True,
            no_qon_async_commands=True,
            qon_log_commands=False,
            qon_close_join_timeout_seconds=1.0,
        )

        adapter = _create_camera_adapter(args)

        self.assertIsInstance(adapter, QonHttpCameraAdapter)
        self.assertEqual(adapter._camera_url, "http://192.0.2.20")

    def test_visca_update_sends_pan_tilt_drive(self):
        adapter, fake_socket = self._visca()

        adapter.update(BoundingBox(150, 80, 170, 100), (100, 200, 3))

        packet, address = fake_socket.sent[-1]
        self.assertEqual(address, ("192.0.2.10", 52381))
        self.assertEqual(packet[:8], bytes.fromhex("01 00 00 09 00 00 00 01"))
        self.assertEqual(packet[8:], bytes.fromhex("81 01 06 01 08 06 02 02 ff"))
        self.assertEqual(adapter.status(), "pan:right tilt:down")

    def test_visca_release_sends_stop(self):
        adapter, fake_socket = self._visca()

        adapter.release()

        self.assertEqual(fake_socket.sent[-1][0][8:], bytes.fromhex("81 01 06 01 08 06 03 03 ff"))

    def test_qon_http_update_sends_observed_ptz_command(self):
        opener = _FakeHttpOpener()
        adapter = QonHttpCameraAdapter(
            "camera",
            min_speed=2,
            max_speed=10,
            deadzone=0.1,
            command_ttl_seconds=0,
            async_commands=False,
            now_fn=lambda: 1.0,
            opener=opener,
        )

        adapter.update(BoundingBox(150, 40, 190, 80), (100, 200, 3))

        request, timeout = opener.requests[-1]
        self.assertEqual(timeout, 1.0)
        self.assertIn("http://camera/cgi-bin/ptzctrl.cgi?ptzcmd&right&", request.full_url)
        self.assertEqual(adapter.status(), "right:7")

    def _tuned_qon_adapter(self, clock, **options):
        return QonHttpCameraAdapter(
            "camera",
            min_speed=2,
            max_speed=10,
            deadzone=0.1,
            stop_deadzone=0.07,
            stop_debounce_seconds=0,
            command_min_interval_seconds=0,
            command_ttl_seconds=10,
            async_commands=False,
            configure_supervisor_actuator=False,
            now_fn=lambda: clock[0],
            opener=_FakeHttpOpener(),
            **options,
        )

    @staticmethod
    def _box_at(error_x):
        # 200x100 frame: half width 100, so centre x = 100 + 100 * error.
        centre = 100 + 100 * error_x
        return BoundingBox(centre - 10, 40, centre + 10, 60)

    def test_qon_http_speed_map_replaces_linear_ramp(self):
        clock = [1.0]
        adapter = self._tuned_qon_adapter(clock, speed_map=[(0.10, 2), (0.14, 3), (0.22, 5)])
        adapter.update(self._box_at(0.25), (100, 200, 3))
        self.assertEqual(adapter.status(), "right:5")
        default = self._tuned_qon_adapter(clock)
        default.update(self._box_at(0.25), (100, 200, 3))
        self.assertEqual(default.status(), "right:3")

    def test_qon_http_prediction_leads_a_moving_target(self):
        clock = [1.0]
        speeds = {2: 43.0, 3: 64.0, 4: 119.0, 5: 189.0, 6: 217.0}
        adapter = self._tuned_qon_adapter(clock, predict_target_seconds=0.4, level_speeds=speeds)
        plain = self._tuned_qon_adapter(clock)
        # Target inside the deadzone but moving right at 0.5 half-frames per second.
        for step in range(6):
            clock[0] = 1.0 + step * 0.1
            box = self._box_at(-0.2 + 0.05 * step)
            adapter.update(box, (100, 200, 3))
            plain.update(box, (100, 200, 3))
        self.assertEqual(plain.status(), "ptzstop")
        self.assertTrue(adapter.status().startswith("right:"))

    def test_qon_http_stop_lookahead_stops_before_the_picture_shows_it(self):
        clock = [1.0]
        speeds = {2: 43.0, 3: 64.0, 4: 119.0, 5: 189.0, 6: 217.0}
        adapter = self._tuned_qon_adapter(clock, stop_lookahead_seconds=0.4, level_speeds=speeds)
        plain = self._tuned_qon_adapter(clock)
        for candidate in (adapter, plain):
            candidate.update(self._box_at(0.2), (100, 200, 3))
            self.assertEqual(candidate.status(), "right:3")
        # 0.4 s later the delayed picture still shows an offset of 0.14, but the
        # camera has been turning at level 3 (64 px/s = 0.2/s) for the whole
        # delay, so 0.08 of it is already covered.
        clock[0] = 1.4
        for candidate in (adapter, plain):
            candidate.update(self._box_at(0.14), (100, 200, 3))
        self.assertTrue(plain.status().startswith("right:"))
        self.assertEqual(adapter.status(), "ptzstop")

    def test_qon_http_diagonal_mode_moves_both_axes_in_one_command(self):
        clock = [1.0]
        adapter = self._tuned_qon_adapter(clock, axis_mode="diagonal")
        # 200x100 frame: 0.3 right of centre and 0.5 below centre.
        adapter.update(BoundingBox(120, 65, 140, 85), (100, 200, 3))
        request, _ = adapter._opener.requests[-1]
        self.assertIn("ptzcmd&rightdown&4&6", request.full_url)
        self.assertEqual(adapter.status(), "rightdown:4/6")
        # Only the pan error remains: back to a single-axis command.
        clock[0] = 1.1
        adapter.update(BoundingBox(120, 40, 140, 60), (100, 200, 3))
        self.assertEqual(adapter.status(), "right:4")

    def test_qon_http_alternate_mode_takes_turns_between_axes(self):
        clock = [0.0]
        adapter = self._tuned_qon_adapter(clock, axis_mode="alternate", axis_alternate_seconds=0.35)
        seen = []
        for step in range(8):
            clock[0] = step * 0.1
            adapter.update(BoundingBox(120, 65, 140, 85), (100, 200, 3))
            seen.append(adapter.status().split(":")[0])
        self.assertEqual(seen, ["right"] * 4 + ["down"] * 3 + ["right"])

    def test_qon_http_dominant_mode_is_unchanged_by_default(self):
        clock = [1.0]
        adapter = self._tuned_qon_adapter(clock)
        adapter.update(BoundingBox(120, 65, 140, 85), (100, 200, 3))
        self.assertEqual(adapter.status(), "down:6")

    def test_qon_http_coasts_in_last_pan_direction_after_loss(self):
        clock = [1.0]
        adapter = self._tuned_qon_adapter(clock, lost_coast_seconds=1.0, lost_coast_speed=2)
        adapter.update(self._box_at(0.5), (100, 200, 3))
        self.assertTrue(adapter.status().startswith("right:"))
        clock[0] = 1.2
        adapter.update(None, (100, 200, 3), immediate_stop=False, allow_coast=True)
        self.assertEqual(adapter.status(), "right:2")
        clock[0] = 1.9
        adapter.release()
        adapter.update(None, (100, 200, 3), immediate_stop=True, allow_coast=True)
        self.assertEqual(adapter.status(), "right:2")
        clock[0] = 2.3
        adapter.update(None, (100, 200, 3), immediate_stop=True, allow_coast=True)
        self.assertEqual(adapter.status(), "ptzstop")

    def test_qon_http_does_not_coast_when_not_allowed_or_disabled(self):
        clock = [1.0]
        for options in ({"lost_coast_seconds": 1.0}, {}):
            adapter = self._tuned_qon_adapter(clock, **options)
            adapter.update(self._box_at(0.5), (100, 200, 3))
            adapter.update(None, (100, 200, 3), immediate_stop=True, allow_coast=not options)
            self.assertEqual(adapter.status(), "ptzstop")

    def test_qon_http_two_axis_tilt_deadzone_and_reverse_hold(self):
        clock = [1.0]
        adapter = self._tuned_qon_adapter(
            clock, axis_mode="diagonal", tilt_deadzone=0.6, tilt_stop_deadzone=0.4,
            axis_reverse_hold_seconds=0.5,
        )
        # Tilt error 0.5 is inside the wider tilt deadzone: pan only.
        adapter.update(BoundingBox(120, 65, 140, 85), (100, 200, 3))
        self.assertEqual(adapter.status(), "right:4")
        # The target jumps to the other side: pan must wait before reversing.
        clock[0] = 1.2
        adapter.update(BoundingBox(60, 40, 80, 60), (100, 200, 3))
        self.assertEqual(adapter.status(), "ptzstop")
        clock[0] = 1.8
        adapter.update(BoundingBox(60, 40, 80, 60), (100, 200, 3))
        self.assertEqual(adapter.status(), "left:4")

    def test_qon_http_zoom_commands_use_single_speed_and_refresh(self):
        clock = [1.0]
        adapter = self._tuned_qon_adapter(clock)
        adapter._command_ttl_seconds = 0.35
        adapter.zoom("in", 1)
        adapter.zoom("in", 1)
        urls = [request.full_url for request, _ in adapter._opener.requests]
        self.assertEqual(urls, ["http://camera/cgi-bin/ptzctrl.cgi?ptzcmd&zoomin&1"])
        clock[0] = 1.4
        adapter.zoom("in", 1)
        adapter.zoom("stop", 0)
        adapter.zoom("stop", 0)
        urls = [request.full_url.rsplit("ptzcmd&", 1)[1] for request, _ in adapter._opener.requests]
        self.assertEqual(urls, ["zoomin&1", "zoomin&1", "zoomstop&0"])

    def test_qon_http_zoom_burst_limit_sends_zoomstop_and_rests(self):
        clock = [1.0]
        adapter = self._tuned_qon_adapter(clock, zoom_max_burst_seconds=0.5, zoom_burst_rest_seconds=0.3)
        adapter._command_ttl_seconds = 0.35
        self.assertEqual(adapter.zoom("out", 4), "out")
        clock[0] = 1.4
        self.assertEqual(adapter.zoom("out", 4), "out")
        clock[0] = 1.5
        self.assertEqual(adapter.zoom("out", 4), "stop")
        clock[0] = 1.7
        self.assertEqual(adapter.zoom("out", 4), "stop")
        clock[0] = 1.85
        self.assertEqual(adapter.zoom("out", 4), "out")
        urls = [request.full_url.rsplit("ptzcmd&", 1)[1] for request, _ in adapter._opener.requests]
        self.assertEqual(urls, ["zoomout&4", "zoomout&4", "zoomstop&0", "zoomout&4"])

    def test_qon_http_failed_zoomstop_is_sent_again(self):
        clock = [1.0]
        adapter = self._tuned_qon_adapter(clock)
        adapter.zoom("out", 4)
        working = adapter._opener

        def broken(request, timeout=None):
            raise OSError("no route")

        adapter._opener = broken
        adapter.zoom("stop", 0)
        adapter._opener = working
        adapter.zoom("stop", 0)
        adapter.zoom("stop", 0)
        urls = [request.full_url.rsplit("ptzcmd&", 1)[1] for request, _ in working.requests]
        self.assertEqual(urls, ["zoomout&4", "zoomstop&0"])

    def test_qon_http_zoom_to_sends_four_hex_digits(self):
        clock = [1.0]
        adapter = self._tuned_qon_adapter(clock)
        adapter.zoom_to(5543, 4)
        adapter.zoom("stop", 0)
        urls = [request.full_url.rsplit("ptzcmd&", 1)[1] for request, _ in adapter._opener.requests]
        self.assertEqual(urls, ["zoomto&4&15A7", "zoomstop&0"])

    def test_qon_http_last_seen_place_and_search_pan(self):
        clock = [1.0]
        speeds = {2: 43.0, 3: 64.0, 4: 119.0, 5: 189.0, 6: 217.0}
        adapter = self._tuned_qon_adapter(clock, predict_target_seconds=0.4, level_speeds=speeds)
        self.assertIsNone(adapter.last_seen_error())
        adapter.update(self._box_at(0.05), (100, 200, 3))
        clock[0] = 2.0
        self.assertAlmostEqual(adapter.last_seen_error()[0], 0.05, delta=0.01)
        adapter.search_pan("right", 4)
        self.assertEqual(adapter.status(), "right:4")
        clock[0] = 3.0
        self.assertLess(adapter.last_seen_error()[0], 0.0)
        adapter.search_pan(None, 4)
        self.assertEqual(adapter.status(), "ptzstop")

    def test_qon_http_view_gain_lowers_the_level_when_zoomed_in(self):
        clock = [1.0]
        adapter = self._tuned_qon_adapter(clock)
        adapter.update(self._box_at(0.5), (100, 200, 3))
        self.assertEqual(adapter.status(), "right:6")
        zoomed = self._tuned_qon_adapter(clock)
        zoomed.set_view_gain(2.0)
        zoomed.update(self._box_at(0.5), (100, 200, 3))
        self.assertEqual(zoomed.status(), "right:3")

    def test_qon_http_follows_prediction_while_target_is_hidden(self):
        clock = [1.0]
        speeds = {2: 43.0, 3: 64.0, 4: 119.0, 5: 189.0, 6: 217.0}
        adapter = self._tuned_qon_adapter(clock, predict_target_seconds=0.4, level_speeds=speeds)
        self.assertIsNone(adapter.predicted_error())
        # Target moving right at 0.5 half-frames per second, last seen at +0.05.
        for step in range(6):
            clock[0] = 1.0 + step * 0.1
            adapter.update(self._box_at(-0.2 + 0.05 * step), (100, 200, 3))
        clock[0] = 2.0
        predicted = adapter.predicted_error()
        self.assertGreater(predicted[0], 0.15)
        self.assertTrue(adapter.follow_prediction())
        self.assertTrue(adapter.status().startswith("right:"))
        # The camera's own commanded motion is taken off the prediction over time.
        clock[0] = 6.0
        self.assertLess(adapter.predicted_error()[0], predicted[0])

    def test_qon_http_prediction_requires_level_speeds(self):
        with self.assertRaises(ValueError):
            QonHttpCameraAdapter("camera", predict_target_seconds=0.4, configure_supervisor_actuator=False)

    def test_qon_http_soft_coast_caps_speed(self):
        opener = _FakeHttpOpener()
        adapter = QonHttpCameraAdapter(
            "camera",
            min_speed=2,
            max_speed=10,
            deadzone=0.1,
            command_ttl_seconds=0,
            async_commands=False,
            now_fn=lambda: 1.0,
            opener=opener,
        )

        adapter.update(
            BoundingBox(150, 40, 190, 80),
            (100, 200, 3),
            speed_limit=2,
        )

        self.assertEqual(adapter.status(), "right:2")

    def test_qon_http_full_speed_error_reaches_max_speed_sooner(self):
        def speed_for(**kwargs):
            adapter = QonHttpCameraAdapter(
                "camera",
                min_speed=2,
                max_speed=10,
                deadzone=0.1,
                configure_supervisor_actuator=False,
                async_commands=False,
                now_fn=lambda: 1.0,
                opener=_FakeHttpOpener(),
                **kwargs,
            )
            # Target centre at x=150 of 200: horizontal error 0.5.
            adapter.update(BoundingBox(140, 40, 160, 60), (100, 200, 3))
            return adapter.status()

        self.assertEqual(speed_for(), "right:6")
        self.assertEqual(speed_for(full_speed_error=0.5), "right:10")

    def test_qon_http_uniform_error_scale_ignores_small_vertical_offsets(self):
        def status_for(**kwargs):
            adapter = QonHttpCameraAdapter(
                "camera",
                deadzone=0.1,
                stop_deadzone=0.07,
                configure_supervisor_actuator=False,
                async_commands=False,
                now_fn=lambda: 1.0,
                opener=_FakeHttpOpener(),
                **kwargs,
            )
            # 640x360 frame, target 25 px below centre: 0.14 of the half
            # height but only 0.08 of the half width.
            adapter.update(BoundingBox(300, 185, 340, 225), (360, 640, 3))
            return adapter.status()

        self.assertEqual(status_for(), "down:2")
        self.assertEqual(status_for(uniform_error_scale=True), "ptzstop")

    def test_qon_http_axis_hysteresis_keeps_the_current_axis(self):
        def commands_for(**kwargs):
            opener = _FakeHttpOpener()
            clock = [1.0]
            adapter = QonHttpCameraAdapter(
                "camera",
                deadzone=0.1,
                stop_deadzone=0.07,
                command_ttl_seconds=10,
                configure_supervisor_actuator=False,
                async_commands=False,
                now_fn=lambda: clock[0],
                opener=opener,
                **kwargs,
            )
            # Horizontal error 0.30; vertical error alternates 0.28 / 0.32.
            for step, center_y in enumerate((64, 66, 64, 66, 64)):
                clock[0] = 1.0 + 0.1 * step
                adapter.update(
                    BoundingBox(120, center_y - 10, 140, center_y + 10),
                    (100, 200, 3),
                )
            return [
                request.full_url.split("ptzcmd&")[1].split("&")[0]
                for request, _ in opener.requests
            ]

        self.assertEqual(commands_for(), ["right", "down", "right", "down", "right"])
        self.assertEqual(commands_for(axis_switch_ratio=1.5), ["right"])

    def test_qon_http_response_delay_compensation_stops_before_the_target(self):
        def status_after_approach(**kwargs):
            clock = [1.0]
            adapter = QonHttpCameraAdapter(
                "camera",
                min_speed=2,
                max_speed=10,
                deadzone=0.1,
                stop_deadzone=0.07,
                stop_debounce_seconds=0,
                command_ttl_seconds=10,
                configure_supervisor_actuator=False,
                async_commands=False,
                now_fn=lambda: clock[0],
                opener=_FakeHttpOpener(),
                **kwargs,
            )
            adapter.update(BoundingBox(125, 40, 145, 60), (100, 200, 3))
            self.assertTrue(adapter.status().startswith("right:"))
            # One second later the picture still shows an error of 0.12, but
            # the motion already commanded will remove it.
            clock[0] = 2.0
            adapter.update(BoundingBox(102, 40, 122, 60), (100, 200, 3))
            return adapter.status()

        self.assertTrue(status_after_approach().startswith("right:"))
        self.assertEqual(
            status_after_approach(response_delay_seconds=0.4),
            "ptzstop",
        )

    def test_qon_http_delay_compensation_never_reverses_direction(self):
        clock = [1.0]
        opener = _FakeHttpOpener()
        adapter = QonHttpCameraAdapter(
            "camera",
            deadzone=0.1,
            stop_deadzone=0.07,
            stop_debounce_seconds=0,
            command_ttl_seconds=10,
            response_delay_seconds=2.0,
            pan_rate_per_speed=1.0,
            configure_supervisor_actuator=False,
            async_commands=False,
            now_fn=lambda: clock[0],
            opener=opener,
        )
        adapter.update(BoundingBox(150, 40, 190, 80), (100, 200, 3))
        clock[0] = 3.0
        adapter.update(BoundingBox(150, 40, 190, 80), (100, 200, 3))

        commands = [
            request.full_url.split("ptzcmd&")[1].split("&")[0]
            for request, _ in opener.requests
        ]
        self.assertNotIn("left", commands)

    def test_qon_http_profile_settings_do_not_hunt_with_delayed_video(self):
        """Closed loop with a delayed picture: the old settings hunt, the profile settles."""

        def reversals(seed, **kwargs):
            import random

            noise = random.Random(seed)
            clock = [0.0]
            sent = []

            class _Recording(QonHttpCameraAdapter):
                def _send(self, command, speed_x, speed_y):
                    del speed_y
                    sent.append((clock[0], command, speed_x))

            adapter = _Recording(
                "camera",
                min_speed=2,
                max_speed=10,
                deadzone=0.10,
                stop_deadzone=0.07,
                stop_debounce_seconds=0.08,
                command_min_interval_seconds=0.08,
                command_ttl_seconds=0.35,
                configure_supervisor_actuator=False,
                async_commands=False,
                now_fn=lambda: clock[0],
                opener=_FakeHttpOpener(),
                **kwargs,
            )
            width, height = 640, 360
            # Offsets in half frame widths / half frame heights, as measured
            # on the camera: about 0.09 and 0.13 per second per speed step,
            # 0.15 s command delay and 0.45 s video delay at 8 fps.
            error_x, error_y = 0.6, -0.32
            history = []
            step, time_now, next_frame = 0.01, 0.0, 0.0
            while time_now < 12.0:
                command, speed = "ptzstop", 0
                for sent_at, sent_command, sent_speed in sent:
                    if sent_at <= time_now - 0.15:
                        command, speed = sent_command, sent_speed
                error_x -= {"right": 1, "left": -1}.get(command, 0) * 0.09 * speed * step
                error_y -= {"down": 1, "up": -1}.get(command, 0) * 0.13 * speed * step
                history.append((error_x, error_y))
                if time_now >= next_frame:
                    next_frame += 0.125
                    seen_x, seen_y = history[max(0, len(history) - 46)]
                    center_x = width / 2 + (seen_x + noise.gauss(0, 0.012)) * width / 2
                    center_y = height / 2 + (seen_y + noise.gauss(0, 0.012)) * height / 2
                    clock[0] = time_now
                    adapter.update(
                        BoundingBox(center_x - 50, center_y - 80, center_x + 50, center_y + 80),
                        (height, width, 3),
                    )
                time_now += step
            count = 0
            last = {}
            for _, command, _ in sent:
                axis = {"left": "x", "right": "x", "up": "y", "down": "y"}.get(command)
                if axis is None:
                    continue
                if axis in last and last[axis] != command:
                    count += 1
                last[axis] = command
            return count, (error_x, error_y)

        profile = dict(
            full_speed_error=0.5,
            uniform_error_scale=True,
            axis_switch_ratio=1.5,
            response_delay_seconds=0.5,
        )
        old_reversals = sum(reversals(seed)[0] for seed in range(5))
        for seed in range(5):
            count, (error_x, error_y) = reversals(seed, **profile)
            # At most one correction, then the camera rests on the target.
            self.assertLessEqual(count, 1)
            self.assertLess(abs(error_x), 0.12)
            # 0.12 half widths expressed in half heights.
            self.assertLess(abs(error_y), 0.12 * 640 / 360)
        # The previous settings keep reversing for the whole run.
        self.assertGreater(old_reversals, 20)

    def test_qon_http_deadzone_hysteresis_and_debounce_avoid_stop_chatter(self):
        opener = _FakeHttpOpener()
        clock = [1.0]
        adapter = QonHttpCameraAdapter(
            "camera",
            deadzone=0.12,
            stop_deadzone=0.08,
            stop_debounce_seconds=0.12,
            command_ttl_seconds=10,
            configure_supervisor_actuator=False,
            async_commands=False,
            now_fn=lambda: clock[0],
            opener=opener,
        )

        adapter.update(BoundingBox(120, 40, 140, 60), (100, 200, 3))
        clock[0] = 1.05
        adapter.update(BoundingBox(99, 40, 119, 60), (100, 200, 3))
        self.assertEqual(adapter.status(), "right:4")

        clock[0] = 1.10
        adapter.update(BoundingBox(95, 40, 105, 60), (100, 200, 3))
        self.assertEqual(adapter.status(), "right:4")
        clock[0] = 1.23
        adapter.update(BoundingBox(95, 40, 105, 60), (100, 200, 3))

        self.assertEqual(adapter.status(), "ptzstop")
        self.assertIn("ptzcmd&ptzstop", opener.requests[-1][0].full_url)

    def test_qon_http_bridges_short_grace_but_safety_stop_is_immediate(self):
        opener = _FakeHttpOpener()
        clock = [1.0]
        adapter = QonHttpCameraAdapter(
            "camera",
            lost_target_hold_seconds=0.2,
            command_ttl_seconds=10,
            configure_supervisor_actuator=False,
            async_commands=False,
            now_fn=lambda: clock[0],
            opener=opener,
        )
        adapter.update(BoundingBox(150, 40, 190, 80), (100, 200, 3))

        clock[0] = 1.1
        adapter.update(None, (100, 200, 3), immediate_stop=False)
        self.assertTrue(adapter.status().startswith("right:"))
        clock[0] = 1.31
        adapter.update(None, (100, 200, 3), immediate_stop=False)
        self.assertEqual(adapter.status(), "ptzstop")

        clock[0] = 2.0
        adapter.update(BoundingBox(150, 40, 190, 80), (100, 200, 3))
        adapter.update(None, (100, 200, 3), immediate_stop=True)
        self.assertEqual(adapter.status(), "ptzstop")

    def test_qon_http_log_commands_logs_successful_request(self):
        opener = _FakeHttpOpener()
        adapter = QonHttpCameraAdapter(
            "camera",
            command_ttl_seconds=0,
            configure_supervisor_actuator=False,
            async_commands=False,
            log_commands=True,
            now_fn=lambda: 1.0,
            opener=opener,
        )

        with self.assertLogs(level="INFO") as logs:
            adapter.update(BoundingBox(150, 40, 190, 80), (100, 200, 3))

        output = "\n".join(logs.output)
        self.assertIn("Qon HTTP PTZ send GET", output)
        self.assertIn("ptzcmd&right&", output)
        self.assertIn("Qon HTTP PTZ response command=right status=200", output)

    def test_qon_http_initializes_supervisor_actuator_mode(self):
        opener = _FakeHttpOpener()

        QonHttpCameraAdapter("camera", async_commands=False, opener=opener)

        urls = [request.full_url for request, _ in opener.requests]
        self.assertEqual(urls[0], "http://camera/cgi-bin/param.cgi?write_path")
        self.assertEqual(urls[1], "http://camera/cgi-bin/param.cgi?post_visca")
        self.assertEqual(urls[2], "http://camera/cgi-bin/param.cgi?get_path")
        write_data = parse_qs(opener.requests[0][0].data.decode())
        self.assertEqual(write_data["common.track"], ["0"])
        self.assertEqual(write_data["tracking.auto_zoom"], ["0"])
        self.assertEqual(write_data["tracking.auto_tilt"], ["0"])
        self.assertEqual(write_data["common.debug_mode"], ["3"])
        refresh_data = parse_qs(opener.requests[1][0].data.decode())
        self.assertEqual(refresh_data["visca"], ["0x81,0x0A,0x01,0x04,0x1D,0x17,0xFF"])

    def test_qon_http_release_sends_stop_after_move(self):
        opener = _FakeHttpOpener()
        adapter = QonHttpCameraAdapter(
            "http://camera",
            command_ttl_seconds=0,
            async_commands=False,
            now_fn=lambda: 1.0,
            opener=opener,
        )

        adapter.update(BoundingBox(150, 40, 190, 80), (100, 200, 3))
        adapter.release()

        self.assertEqual(
            opener.requests[-1][0].full_url,
            "http://camera/cgi-bin/ptzctrl.cgi?ptzcmd&ptzstop&10&10",
        )

    def test_qon_http_basic_auth_adds_authorization_header(self):
        opener = _FakeHttpOpener()
        adapter = QonHttpCameraAdapter(
            "http://camera",
            username="user",
            password="pass",
            auth_mode="basic",
            async_commands=False,
            configure_supervisor_actuator=False,
            opener=opener,
        )

        adapter.update(BoundingBox(150, 40, 190, 80), (100, 200, 3))

        self.assertEqual(opener.requests[-1][0].get_header("Authorization"), "Basic dXNlcjpwYXNz")

    def test_qon_http_async_close_sends_synchronous_stop(self):
        opener = _FakeHttpOpener()
        adapter = QonHttpCameraAdapter(
            "http://camera",
            configure_supervisor_actuator=False,
            async_commands=True,
            close_join_timeout_seconds=0.1,
            opener=opener,
        )

        adapter.update(BoundingBox(150, 40, 190, 80), (100, 200, 3))
        adapter.close()

        self.assertEqual(
            opener.requests[-1][0].full_url,
            "http://camera/cgi-bin/ptzctrl.cgi?ptzcmd&ptzstop&10&10",
        )


class ArucoFlipModeTest(unittest.TestCase):
    def setUp(self):
        import cv2
        import numpy as np

        from marker_tracker import app

        self.cv2, self.app = cv2, app
        self.detector = app._create_aruco_detector(cv2, "DICT_4X4_50")
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        marker = cv2.aruco.generateImageMarker(dictionary, 0, 80)
        frame = np.full((360, 640), 255, dtype=np.uint8)
        frame[100:180, 400:480] = marker
        self.frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

    def _detect(self, frame, flip_mode):
        return self.app._detect_aruco_markers(
            self.cv2, self.detector, frame, marker_id=0, flip_mode=flip_mode
        )

    def test_unmirrored_tries_only_the_frame_and_its_rotation(self):
        attempts = self.app._aruco_detection_attempts(self.cv2, self.frame, "unmirrored")
        self.assertEqual([mode for _, mode in attempts], ["none", "both"])

    def test_unmirrored_finds_an_upright_marker_at_its_position(self):
        detections = self._detect(self.frame, "unmirrored")
        self.assertEqual(len(detections), 1)
        self.assertAlmostEqual(detections[0].box.x1, 400, delta=2)
        self.assertAlmostEqual(detections[0].box.y1, 100, delta=2)

    def test_mirrored_stream_needs_auto(self):
        mirrored = self.cv2.flip(self.frame, 1)
        self.assertEqual(self._detect(mirrored, "unmirrored"), [])
        self.assertEqual(len(self._detect(mirrored, "auto")), 1)


class SpeakerReidHefOptionTest(unittest.TestCase):
    def test_npu_reid_requires_explicit_similarity_calibration(self):
        from marker_tracker import app

        args = app.parse_args(
            [
                "--video-file", "x.mp4",
                "--speaker-identity-backend", "yolo-reid",
                "--speaker-reid-hef", "repvgg_a0_person_reid_512_h8.hef",
            ]
        )
        with self.assertRaises(SystemExit):
            app._create_speaker_identity_manager(args)


class ParallelTargetSignalTest(ArucoFlipModeTest):
    def test_frame_only_source_gives_the_same_markers_from_a_worker_thread(self):
        from concurrent.futures import ThreadPoolExecutor

        source = self.app.ArucoTargetSource(self.cv2, "DICT_4X4_50", 0, "unmirrored")
        self.assertTrue(source.frame_only)
        direct = source.update(self.frame, None, []).detections
        with ThreadPoolExecutor(max_workers=1) as executor:
            result, seconds = executor.submit(
                self.app._timed_target_signal, source, self.frame
            ).result()
        self.assertEqual(result.detections, direct)
        self.assertEqual(len(direct), 1)
        self.assertGreater(seconds, 0.0)

    def test_sources_that_read_the_person_result_stay_serial(self):
        self.assertFalse(getattr(self.app.ManualTargetSource, "frame_only", False))


class OverlayTest(unittest.TestCase):
    class RecordingCv2:
        FONT_HERSHEY_SIMPLEX = 0
        MARKER_CROSS = 0

        def __init__(self):
            self.texts, self.rectangles, self.markers = [], [], 0

        def putText(self, frame, text, *args):
            self.texts.append(text)

        def rectangle(self, frame, first, second, color, thickness):
            self.rectangles.append((color, thickness))

        def drawMarker(self, *args, **kwargs):
            self.markers += 1

    def _draw(self, state, overlay, target=True):
        from marker_tracker import app

        cv2 = self.RecordingCv2()
        person = Detection("person", 0.9, BoundingBox(100, 50, 200, 300), track_id=7)
        marker = Detection("marker", 1.0, BoundingBox(140, 150, 160, 170))
        decision = HandoffDecision(state, person.box if target else None, marker.box, 7)
        app._draw(
            cv2, None, [person, marker], decision, decision.target_box,
            person.box if target else None, 24.4, "aruco", "upper-body", "left:3",
            overlay=overlay,
        )
        return cv2

    def test_simple_overlay_shows_only_fps_and_one_state_word(self):
        cv2 = self._draw("tracking", "simple")
        self.assertEqual(sorted(set(cv2.texts)), ["FPS 24", "TRACKING"])
        self.assertIn(((0, 255, 0), 3), cv2.rectangles)  # speaker: thick green
        self.assertIn(((255, 160, 0), 1), cv2.rectangles)  # other people: thin
        self.assertIn(((0, 255, 255), 2), cv2.rectangles)  # marker
        self.assertEqual(cv2.markers, 1)  # camera aim cross kept

    def test_simple_overlay_state_words(self):
        words = {
            "default": "WAITING", "confirming": "REGISTERING", "tracking": "TRACKING",
            "identity_resolving": "SEARCHING", "recovering": "SEARCHING", "grace": "SEARCHING",
        }
        for state, word in words.items():
            self.assertIn(word, self._draw(state, "simple", target=False).texts)

    def test_debug_overlay_keeps_labels_and_ids(self):
        texts = self._draw("tracking", "debug").texts
        self.assertTrue(any("person" in text and "id:7" in text for text in texts))
        self.assertTrue(any(text.startswith("reason:") for text in texts))


class OverlayStateWordTest(unittest.TestCase):
    def test_one_frame_dropout_keeps_tracking_but_a_real_loss_shows_searching(self):
        from marker_tracker import app

        clock = [0.0]
        words = app.OverlayStateWord(hold_seconds=0.4, now_fn=lambda: clock[0])
        self.assertEqual(words.word("confirming")[0], "REGISTERING")
        self.assertEqual(words.word("tracking")[0], "TRACKING")
        clock[0] = 1.00
        self.assertEqual(words.word("grace")[0], "TRACKING")
        clock[0] = 1.05
        self.assertEqual(words.word("tracking")[0], "TRACKING")
        clock[0] = 2.00
        self.assertEqual(words.word("identity_resolving")[0], "TRACKING")
        clock[0] = 2.50
        self.assertEqual(words.word("identity_resolving")[0], "SEARCHING")
        clock[0] = 2.55
        self.assertEqual(words.word("default")[0], "WAITING")


if __name__ == "__main__":
    unittest.main()


class LostSearchTest(unittest.TestCase):
    class _Camera:
        def __init__(self, predicted, last_seen):
            self._predicted, self._last_seen = predicted, last_seen

        def predicted_error(self, now=None):
            return self._predicted

        def last_seen_error(self, now=None):
            return self._last_seen

    @staticmethod
    def _decision(state, target=None):
        from types import SimpleNamespace

        return SimpleNamespace(state=state, target_box=target)

    def _lost(self, **options):
        from marker_tracker.app import LostSearch

        search = LostSearch(wait_seconds=2.0, sweep_seconds=6.0, pause_seconds=3.0, **options)
        camera = self._Camera((0.6, 0.0), (0.4, 0.0))
        target = BoundingBox(400, 100, 440, 300)
        self.assertIsNone(search.update(self._decision("tracking", target), [], 0.0, camera).candidate)
        return search, camera

    def test_nothing_before_a_target_was_tracked(self):
        from marker_tracker.app import LostSearch

        person = Detection("person", 0.8, BoundingBox(100, 100, 140, 300), track_id=3)
        plan = LostSearch().update(self._decision("default"), [person], 0.0, self._Camera(None, None))
        self.assertIsNone(plan.candidate)
        self.assertFalse(plan.active)

    def test_person_in_view_is_steered_to(self):
        search, camera = self._lost()
        near = Detection("person", 0.5, BoundingBox(380, 100, 420, 300), track_id=4)
        far = Detection("person", 0.9, BoundingBox(50, 100, 90, 300), track_id=5)
        plan = search.update(self._decision("default"), [near, far], 1.0, camera)
        self.assertEqual(plan.candidate, near.box)
        self.assertFalse(plan.active)

    def test_without_zoom_it_looks_to_the_last_seen_side_and_comes_back(self):
        search, camera = self._lost()
        lost = self._decision("default")
        self.assertFalse(search.update(lost, [], 1.0, camera).active)
        plan = search.update(lost, [], 3.5, camera)
        self.assertEqual((plan.active, plan.pan), (True, "right"))
        self.assertEqual(search.update(lost, [], 10.0, camera).pan, None)
        self.assertEqual(search.update(lost, [], 13.0, camera).pan, "left")
        self.assertEqual(search.update(lost, [], 19.0, camera).pan, None)
        self.assertFalse(search.update(lost, [], 70.0, camera).active)
        self.assertFalse(search.update(self._decision("tracking", BoundingBox(1, 1, 9, 9)), [], 71.0, camera).active)

    def test_with_zoom_it_points_at_the_last_seen_place_and_narrows(self):
        search, camera = self._lost()
        lost = self._decision("default")
        search.update(lost, [], 1.0, camera, scale=1.0, zoomable=True)
        plan = search.update(lost, [], 3.5, camera, scale=1.0, zoomable=True)
        self.assertEqual((plan.active, plan.pan, plan.narrow), (True, "right", True))
        camera._last_seen = (0.05, 0.0)
        plan = search.update(lost, [], 5.0, camera, scale=1.0, zoomable=True)
        self.assertEqual((plan.pan, plan.narrow), (None, True))
        plan = search.update(lost, [], 10.0, camera, scale=1.0, zoomable=True)
        self.assertEqual((plan.active, plan.narrow), (False, False))

    def test_candidates_only_when_the_sweep_is_off(self):
        search, camera = self._lost(sweep=False)
        self.assertFalse(search.update(self._decision("default"), [], 30.0, camera).active)
