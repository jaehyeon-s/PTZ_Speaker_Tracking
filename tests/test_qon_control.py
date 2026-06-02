import unittest
from urllib.parse import parse_qs

from src.camera.qon_control import QonTrackingControl, TrackingMode, parse_mode


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


if __name__ == "__main__":
    unittest.main()
