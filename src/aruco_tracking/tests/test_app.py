import unittest
from argparse import Namespace
from tempfile import TemporaryDirectory
from urllib.parse import parse_qs

from marker_tracker.app import (
    ByteTrackDiagnostics,
    FramingPolicy,
    LoggingCameraAdapter,
    NoopReIdentifier,
    PredictIouPersonSource,
    ScoreDiagnostics,
    SimpleIouTracker,
    TrackingStatusDiagnostics,
    YoloPersonSource,
    _create_camera_adapter,
    _create_reidentifier,
    _crop_frame,
    _apply_runtime_profile,
    _read_exported_imgsz,
    _recovery_status,
    _resolve_tracker_config,
    _to_marker_detections,
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


class YoloPersonSourceTest(unittest.TestCase):
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


class YoloClassMarkerTest(unittest.TestCase):
    def test_yolo_class_marker_source_converts_selected_class_to_marker(self):
        detections = _to_marker_detections(
            {0: "person", 1: "badge"},
            [
                _FakeBox(0, 0.95, [0, 0, 100, 200], track_id=3),
                _FakeBox(1, 0.8, [20, 30, 50, 70], track_id=9),
            ],
            marker_class="badge",
            marker_confidence=0.5,
        )

        self.assertEqual(len(detections), 1)
        self.assertEqual(detections[0].label, "marker")
        self.assertEqual(detections[0].track_id, 9)
        self.assertEqual(detections[0].box.center, (35.0, 50.0))

    def test_yolo_class_marker_source_respects_confidence_threshold(self):
        detections = _to_marker_detections(
            {1: "badge"},
            [_FakeBox(1, 0.2, [20, 30, 50, 70])],
            marker_class="badge",
            marker_confidence=0.5,
        )

        self.assertEqual(detections, [])


class ByteTrackDiagnosticsTest(unittest.TestCase):
    def test_logs_recommended_track_buffer_from_fps(self):
        now = iter([10.0, 13.0])
        diagnostics = ByteTrackDiagnostics(
            expected_buffer_seconds=3.0,
            log_interval_seconds=2.0,
            now_fn=lambda: next(now),
        )
        detections = _to_marker_detections(
            {1: "badge"},
            [_FakeBox(1, 0.8, [20, 30, 50, 70], track_id=4)],
            marker_class="badge",
            marker_confidence=0.5,
        )
        person = _FakeBox(0, 0.9, [0, 0, 100, 200], track_id=7)
        person_detection = _to_marker_detections(
            {0: "person"},
            [person],
            marker_class="person",
            marker_confidence=0.5,
        )[0]
        person_detection = person_detection.__class__(
            "person",
            person_detection.confidence,
            person_detection.box,
            person_detection.track_id,
        )

        diagnostics.update([person_detection, *detections], fps=25.0, target_track_id=7)
        with self.assertLogs(level="INFO") as logs:
            diagnostics.update(
                [person_detection, *detections],
                fps=25.0,
                target_track_id=7,
            )

        self.assertIn("recommended_track_buffer=75", "\n".join(logs.output))


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
        self.assertEqual(profiled.bytetrack_config, "bytetrack-pi.yaml")
        self.assertEqual(profiled.aruco_marker_id, 0)
        self.assertEqual(profiled.aruco_detect_flip, "auto")
        self.assertEqual(profiled.confirmation_seconds, 2.0)
        self.assertEqual(profiled.marker_dropout_seconds, 0.75)
        self.assertTrue(profiled.tracking_status_diagnostics)
        self.assertEqual(profiled.framing_mode, "upper-body")
        self.assertEqual(profiled.camera_control, "qon-http")

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
            provided_dests={"model", "confirmation_seconds"},
        )

        self.assertEqual(profiled.model, "custom.pt")
        self.assertEqual(profiled.confirmation_seconds, 1.5)
        self.assertEqual(profiled.imgsz, 416)
        self.assertEqual(profiled.camera_control, "qon-http")


class ModelMetadataTest(unittest.TestCase):
    def test_reads_exported_imgsz_from_ncnn_metadata(self):
        with TemporaryDirectory() as directory:
            metadata_path = f"{directory}/metadata.yaml"
            with open(metadata_path, "w", encoding="utf-8") as handle:
                handle.write("imgsz:\n- 640\n- 640\n")

            self.assertEqual(_read_exported_imgsz(directory), (640, 640))


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


class FramingPolicyTest(unittest.TestCase):
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


class ReIdentifierTest(unittest.TestCase):
    def test_default_reidentifier_is_noop(self):
        reidentifier = _create_reidentifier(Namespace(reid_backend="none"))

        self.assertIsInstance(reidentifier, NoopReIdentifier)

    def test_crop_frame_clips_box_to_frame(self):
        crop_key = _crop_frame(_FakeFrame(), BoundingBox(-10, 20, 250, 80))

        self.assertEqual(crop_key, (slice(20, 80, None), slice(0, 200, None)))

    def test_crop_frame_rejects_empty_box(self):
        crop = _crop_frame(_FakeFrame(), BoundingBox(50, 20, 50, 80))

        self.assertIsNone(crop)


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


if __name__ == "__main__":
    unittest.main()
