import threading
import unittest
from urllib.parse import parse_qs

from src.camera.qon_control import QonTrackingControl, QonVelocityPTZController, TrackingMode, parse_mode


class FakeResponse:
    status = 200

    def __init__(self, body):
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        return False

    def read(self):
        return self.body.encode("utf-8")


class FakeOpener:
    def __init__(self, body="common.track=1"):
        self.body = body
        self.requests = []

    def __call__(self, request, timeout):
        self.requests.append((request, timeout))
        return FakeResponse(self.body)


class QonTrackingControlTests(unittest.TestCase):
    def test_parses_observed_tracking_modes(self):
        self.assertEqual(parse_mode("common.track=1"), TrackingMode.TRACKING)
        self.assertEqual(parse_mode("common.track = 0\n"), TrackingMode.ZONE)
        self.assertIsNone(parse_mode("other.value=1"))

    def test_reads_track_conf_from_observed_endpoint(self):
        opener = FakeOpener("common.track=1")
        control = QonTrackingControl("http://camera", opener=opener)

        mode = control.get_mode()
        request, timeout = opener.requests[0]

        self.assertEqual(mode, TrackingMode.TRACKING)
        self.assertEqual(request.full_url, "http://camera/cgi-bin/param.cgi?get_path")
        self.assertEqual(parse_qs(request.data.decode()), {"path": ["/data/track.conf"]})
        self.assertEqual(timeout, 5.0)

    def test_sets_tracking_or_zone_by_writing_common_track_only(self):
        opener = FakeOpener()
        control = QonTrackingControl("http://camera", opener=opener)

        control.enable_zone_tracking()
        zone_request, _ = opener.requests[-1]
        control.enable_auto_tracking()
        tracking_request, _ = opener.requests[-1]

        self.assertEqual(zone_request.full_url, "http://camera/cgi-bin/param.cgi?write_path")
        self.assertEqual(parse_qs(zone_request.data.decode())["common.track"], ["0"])
        self.assertEqual(parse_qs(tracking_request.data.decode())["common.track"], ["1"])

    def test_sets_observed_debug_and_hint_flags(self):
        opener = FakeOpener()
        control = QonTrackingControl("http://camera", opener=opener)

        control.set_humanoid_frame(2)
        debug_request, _ = opener.requests[-1]
        control.set_tracking_hint(False)
        hint_request, _ = opener.requests[-1]

        self.assertEqual(parse_qs(debug_request.data.decode())["common.debug_mode"], ["2"])
        self.assertEqual(parse_qs(hint_request.data.decode())["common.osd_mode"], ["0"])

    def test_sends_observed_ptz_stop_and_zoom_commands(self):
        opener = FakeOpener()
        control = QonTrackingControl("http://camera", opener=opener)

        control.stop()
        stop_request, _ = opener.requests[-1]
        control.zoom_out()
        zoom_request, _ = opener.requests[-1]

        self.assertEqual(stop_request.full_url, "http://camera/cgi-bin/ptzctrl.cgi?ptzcmd&ptzstop&10&10")
        self.assertEqual(zoom_request.full_url, "http://camera/cgi-bin/ptzctrl.cgi?ptzcmd&zoomout&5")

    def test_supervisor_actuator_mode_uses_write_refresh_readback_sequence(self):
        opener = FakeOpener(
            "common.track=\"0\"\ntracking.auto_zoom=\"0\"\ntracking.auto_tilt=\"0\"\ncommon.debug_mode=\"3\""
        )
        control = QonTrackingControl("http://camera", opener=opener)

        response = control.set_supervisor_actuator_mode()
        urls = [request.full_url for request, _ in opener.requests]

        self.assertEqual(response.body, opener.body)
        self.assertEqual(urls[0], "http://camera/cgi-bin/param.cgi?write_path")
        self.assertEqual(urls[1], "http://camera/cgi-bin/param.cgi?post_visca")
        self.assertEqual(urls[2], "http://camera/cgi-bin/param.cgi?get_path")
        write_data = parse_qs(opener.requests[0][0].data.decode())
        self.assertEqual(write_data["common.track"], ["0"])
        self.assertEqual(write_data["tracking.auto_zoom"], ["0"])
        self.assertEqual(write_data["tracking.auto_tilt"], ["0"])
        self.assertEqual(write_data["common.debug_mode"], ["3"])

    def test_velocity_ptz_sends_direction_then_stop(self):
        opener = FakeOpener()
        control = QonTrackingControl("http://camera", opener=opener)
        ptz = QonVelocityPTZController(control, dead_zone_ratio=0.1, min_speed=2, max_speed=10)

        action = ptz.follow_bbox((150, 40, 190, 80), (100, 200, 3))
        move_url = opener.requests[-1][0].full_url
        ptz.stop()
        stop_url = opener.requests[-1][0].full_url

        self.assertTrue(action.startswith("right:"))
        self.assertIn("ptzcmd&right&", move_url)
        self.assertEqual(stop_url, "http://camera/cgi-bin/ptzctrl.cgi?ptzcmd&ptzstop&10&10")

    def test_velocity_ptz_zooms_in_when_centered_subject_too_small(self):
        opener = FakeOpener()
        control = QonTrackingControl("http://camera", opener=opener)
        ptz = QonVelocityPTZController(
            control,
            dead_zone_ratio=0.2,
            zoom_enabled=True,
            target_height_ratio=0.6,
            zoom_speed=3,
        )

        action = ptz.follow_bbox((90, 45, 110, 55), (100, 200, 3))
        zoom_url = opener.requests[-1][0].full_url

        self.assertTrue(action.startswith("zoomin:"))
        self.assertIn("ptzcmd&zoomin&3", zoom_url)

    def test_velocity_ptz_does_not_hunt_zoom_within_hysteresis_band(self):
        opener = FakeOpener()
        control = QonTrackingControl("http://camera", opener=opener)
        ptz = QonVelocityPTZController(
            control,
            dead_zone_ratio=0.2,
            zoom_enabled=True,
            target_height_ratio=0.6,
            zoom_tolerance=0.05,
            zoom_hysteresis=0.1,
            zoom_smoothing=1.0,  # no EMA lag, isolate the hysteresis logic
            zoom_speed=3,
        )
        frame = (100, 200, 3)

        # Subject exactly on target -> settles, no zoom.
        self.assertEqual(ptz.follow_bbox((90, 20, 110, 80), frame), "ptzstop")
        # Small wobble inside the outer band (height 0.62, 0.58) must NOT zoom.
        self.assertEqual(ptz.follow_bbox((90, 20, 110, 82), frame), "ptzstop")
        self.assertEqual(ptz.follow_bbox((90, 21, 110, 79), frame), "ptzstop")
        # A real drift past the outer band (height 0.76 > 0.70) resumes zooming out.
        action = ptz.follow_bbox((90, 12, 110, 88), frame)
        self.assertTrue(action.startswith("zoomout:"))

    def test_velocity_ptz_aims_up_for_face_when_vertical_aim_is_high(self):
        opener = FakeOpener()
        control = QonTrackingControl("http://camera", opener=opener)
        # Subject centered horizontally, vertically centered box; aim 0.3 -> camera tilts up.
        ptz = QonVelocityPTZController(
            control, dead_zone_ratio=0.05, vertical_aim_ratio=0.3
        )

        action = ptz.follow_bbox((95, 10, 105, 90), (100, 200, 3))

        self.assertTrue(action.startswith("up:"))

    def test_velocity_ptz_skips_zoom_when_disabled(self):
        opener = FakeOpener()
        control = QonTrackingControl("http://camera", opener=opener)
        ptz = QonVelocityPTZController(control, dead_zone_ratio=0.2, zoom_enabled=False)

        action = ptz.follow_bbox((90, 45, 110, 55), (100, 200, 3))

        self.assertEqual(action, "ptzstop")

    def test_async_velocity_ptz_closes_with_stop_command(self):
        class RecordingControl:
            def __init__(self):
                self.calls = []

            def ptz_command(self, command, speed_x, speed_y):
                self.calls.append((command, speed_x, speed_y))

            def stop(self):
                self.ptz_command("ptzstop", 10, 10)

        control = RecordingControl()
        ptz = QonVelocityPTZController(
            control,
            dead_zone_ratio=0.1,
            min_speed=2,
            max_speed=10,
            async_commands=True,
        )

        action = ptz.follow_bbox((150, 40, 190, 80), (100, 200, 3))
        ptz.close()

        self.assertTrue(action.startswith("right:"))
        self.assertGreaterEqual(len(control.calls), 1)
        self.assertEqual(control.calls[-1], ("ptzstop", 10, 10))

    def test_async_velocity_ptz_close_sends_synchronous_stop_if_worker_is_blocked(self):
        class BlockingMoveControl:
            def __init__(self):
                self.calls = []
                self.move_started = threading.Event()

            def ptz_command(self, command, speed_x, speed_y):
                self.calls.append((command, speed_x, speed_y))
                if command != "ptzstop":
                    self.move_started.set()
                    threading.Event().wait(0.2)

            def stop(self):
                self.calls.append(("sync-stop", 10, 10))

        control = BlockingMoveControl()
        ptz = QonVelocityPTZController(
            control,
            dead_zone_ratio=0.1,
            min_speed=2,
            max_speed=10,
            async_commands=True,
            close_join_timeout_seconds=0.01,
        )

        ptz.follow_bbox((150, 40, 190, 80), (100, 200, 3))
        self.assertTrue(control.move_started.wait(0.1))
        ptz.close()

        self.assertIn(("sync-stop", 10, 10), control.calls)


if __name__ == "__main__":
    unittest.main()
