from __future__ import annotations

import unittest

from marker_tracker.core import BoundingBox, Detection, MarkerTracker
from marker_tracker.zoom import ZoomController, ZoomFovTable, ZoomState

TABLE = ZoomFovTable([(0, 60.7), (5543, 20.0), (9000, 10.0), (16384, 2.28)])
FRAME = (360, 640, 3)


def box(centre_x=320.0, top=80.0, height=200.0, width=70.0):
    return BoundingBox(centre_x - width / 2, top, centre_x + width / 2, top + height)


class ZoomFovTableTest(unittest.TestCase):
    def test_round_trip_and_clamping(self):
        self.assertAlmostEqual(TABLE.hfov(5543), 20.0)
        self.assertAlmostEqual(TABLE.position(TABLE.hfov(7000)), 7000, delta=1)
        self.assertAlmostEqual(TABLE.hfov(-5), 60.7)
        self.assertEqual(TABLE.position(1.0), 16384)


class ZoomControllerTest(unittest.TestCase):
    def controller(self, **options):
        return ZoomController(TABLE, **options)

    def settle(self, controller, hfov, target, start=0.0, seconds=3.0, **kwargs):
        action = ("stop", 0)
        for step in range(int(seconds * 10)):
            action = controller.update(start + step * 0.1, hfov, target, FRAME, **kwargs)
        return action

    def test_small_slow_centred_target_zooms_in_slowly(self):
        controller = self.controller()
        # 200 px of 360 at 20 degrees: upper body ratio 0.28, wanted 0.55.
        action = self.settle(controller, 20.0, box(), confidence=0.85)
        self.assertEqual(action, ("in", 1))
        self.assertAlmostEqual(controller.last_ratio, 0.278, delta=0.01)

    def test_no_zoom_in_when_off_centre_fast_or_hidden(self):
        self.assertEqual(self.settle(self.controller(), 20.0, box(centre_x=420), confidence=0.85)[0], "stop")
        fast = self.controller()
        # One half-width per second at 9 degrees is 4.5 deg/s: needs about 14.5 degrees.
        self.assertEqual(self.settle(fast, 9.0, box(height=150), confidence=0.85, target_velocity=1.0)[0], "out")
        hidden = self.controller()
        self.settle(hidden, 20.0, box(), confidence=0.85)
        self.assertEqual(hidden.update(3.1, 20.0, box(height=60), FRAME, confidence=0.4, occluded=True), ("stop", 0))

    def test_zoom_out_at_once_near_edge_or_on_weak_detection(self):
        edge = self.controller()
        self.assertEqual(edge.update(0.0, 12.0, box(centre_x=560), FRAME, confidence=0.85), ("out", 4))
        weak = self.controller()
        self.settle(weak, 12.0, box(height=250), confidence=0.85)
        self.assertEqual(weak.update(3.1, 12.0, box(height=250), FRAME, confidence=0.30), ("out", 4))

    def test_cooldown_after_zoom_movement(self):
        controller = self.controller(cooldown_seconds=2.0)
        self.assertEqual(self.settle(controller, 20.0, box(), confidence=0.85)[0], "in")
        # The edge interrupts the zoom-in; back at the centre it must wait.
        self.assertEqual(controller.update(3.0, 20.0, box(centre_x=560), FRAME, confidence=0.85)[0], "out")
        self.assertEqual(controller.update(3.5, 20.0, box(), FRAME, confidence=0.85)[0], "stop")
        self.assertEqual(controller.update(5.2, 20.0, box(), FRAME, confidence=0.85)[0], "in")

    def test_zoom_in_stops_inside_the_allowed_range(self):
        controller = self.controller()
        # The same person seen at 10.5 degrees fills 200 * 20 / 10.5 = 381 px: ratio 0.53.
        self.settle(controller, 20.0, box(), confidence=0.85)
        clipped = BoundingBox(250, 20, 390, 360)
        self.assertEqual(controller.update(3.1, 10.5, clipped, FRAME, confidence=0.85)[0], "stop")

    def test_lost_target_widens_to_the_configured_field_of_view(self):
        controller = self.controller(lost_hfov=40.0)
        self.assertEqual(controller.update(0.0, 20.0, None, FRAME, lost=True), ("out", 4))
        self.assertEqual(controller.update(0.1, 40.5, None, FRAME, lost=True), ("stop", 0))

    def test_marker_size_tracks_distance_when_feet_are_out_of_frame(self):
        controller = self.controller()
        marker = BoundingBox(300, 150, 320, 170)
        self.settle(controller, 20.0, box(), confidence=0.85, marker_box=marker)
        before = controller.person_degrees()
        # Person walks closer: the box is cut at the bottom, the marker doubles.
        near = BoundingBox(250, 30, 390, 360)
        for step in range(30):
            controller.update(3.0 + step * 0.1, 20.0, near, FRAME, confidence=0.85,
                              marker_box=BoundingBox(300, 150, 340, 190))
        self.assertAlmostEqual(controller.person_degrees() / before, 2.0, delta=0.15)


class ZoomSearchAndLimitsTest(unittest.TestCase):
    def test_lost_with_candidate_narrows_until_marker_is_large_enough(self):
        controller = ZoomController(TABLE, lost_hfov=40.0, search_marker_pixels=14.0)
        candidate = BoundingBox(300, 120, 340, 230)
        small_marker = BoundingBox(315, 160, 323, 168)
        self.assertEqual(controller.update(0.0, 40.0, None, FRAME, lost=True), ("stop", 0))
        self.assertEqual(
            controller.update(0.1, 60.0, None, FRAME, lost=True, search_box=candidate, marker_box=small_marker),
            ("in", 2),
        )
        big_marker = BoundingBox(312, 160, 327, 175)
        self.assertEqual(
            controller.update(0.2, 30.0, None, FRAME, lost=True, search_box=candidate, marker_box=big_marker)[0],
            "stop",
        )
        off_centre = BoundingBox(520, 120, 560, 230)
        self.assertEqual(
            controller.update(0.3, 60.0, None, FRAME, lost=True, search_box=off_centre)[0], "stop"
        )

    def test_ceiling_factor_scales_the_minimum_field_of_view(self):
        results = {}
        for factor in (1.0, 1.5):
            controller = ZoomController(TABLE, ceiling_factor=factor)
            for step in range(20):
                controller.update(step * 0.1, 20.0, box(), FRAME, confidence=0.85, target_velocity=0.5)
            results[factor] = controller.last_min_hfov
        # 0.5 half-widths/s at 20 degrees = 5 deg/s; half width 1.09 degrees.
        self.assertAlmostEqual(results[1.0], 2 * (5.0 + 1.09), delta=0.1)
        self.assertAlmostEqual(results[1.5], 2 * (7.5 + 1.09), delta=0.1)

    def test_zoom_in_waits_while_target_crosses_the_frame_quickly(self):
        # 0.3 half-widths/s = 15% of the frame width per second: above the 10% limit.
        moving = ZoomController(TABLE, delay_seconds=0.1, response_seconds=0.1)
        for step in range(30):
            action = moving.update(step * 0.1, 40.0, box(top=110, height=120, width=50), FRAME, confidence=0.85, target_velocity=0.3)
        self.assertEqual(action[0], "stop")
        calm = ZoomController(TABLE, delay_seconds=0.1, response_seconds=0.1)
        for step in range(30):
            action = calm.update(step * 0.1, 40.0, box(top=110, height=120, width=50), FRAME, confidence=0.85, target_velocity=0.1)
        self.assertEqual(action[0], "in")


class ZoomRunawayGuardTest(unittest.TestCase):
    def test_partial_box_does_not_shrink_the_size_estimate(self):
        controller = ZoomController(TABLE)
        for step in range(30):
            controller.update(step * 0.1, 20.0, box(), FRAME, confidence=0.85)
        whole = controller.person_degrees()
        # Zoomed in, the detector returns a head-and-shoulders box that is not cut by the frame.
        head = BoundingBox(280, 100, 360, 190)
        for step in range(30):
            action = controller.update(3.0 + step * 0.1, 8.0, head, FRAME, confidence=0.85)
        self.assertAlmostEqual(controller.person_degrees(), whole, delta=0.01)
        self.assertNotEqual(action[0], "in")

    def test_large_marker_or_full_frame_box_blocks_zoom_in(self):
        controller = ZoomController(TABLE)
        for step in range(30):
            controller.update(step * 0.1, 20.0, box(), FRAME, confidence=0.85)
        big_marker = BoundingBox(290, 150, 350, 210)
        self.assertEqual(controller.update(3.1, 20.0, box(), FRAME, confidence=0.85, marker_box=big_marker)[0], "out")
        full = BoundingBox(200, 0, 440, 360)
        self.assertEqual(controller.update(3.2, 20.0, full, FRAME, confidence=0.85)[0], "out")


class MarkerSizeReferenceTest(unittest.TestCase):
    def test_large_marker_sets_the_wanted_field_of_view(self):
        controller = ZoomController(TABLE, upper_body_meters=0.81, marker_meters=0.093)
        # A 16 px marker means an upper body of 16 * 0.81 / 0.093 = 139 px: ratio 0.39.
        marker = BoundingBox(312, 150, 328, 166)
        partial = BoundingBox(290, 120, 350, 220)  # not a complete standing box
        for step in range(30):
            action = controller.update(step * 0.1, 20.0, partial, FRAME, confidence=0.85, marker_box=marker)
        self.assertEqual(controller.last_size_source, "marker")
        self.assertAlmostEqual(controller.last_ratio, 0.387, delta=0.01)
        self.assertEqual(action[0], "in")

    def test_small_marker_falls_back_to_learned_box_or_holds(self):
        small = BoundingBox(315, 150, 325, 160)
        learned = ZoomController(TABLE, upper_body_meters=0.81)
        for step in range(30):
            learned.update(step * 0.1, 20.0, box(), FRAME, confidence=0.85, marker_box=small)
        self.assertEqual(learned.last_size_source, "box")
        unknown = ZoomController(TABLE, upper_body_meters=0.81)
        partial = BoundingBox(290, 90, 350, 190)
        self.assertEqual(unknown.update(0.0, 20.0, partial, FRAME, confidence=0.85, marker_box=small), ("stop", 0))
        self.assertEqual(unknown.last_reason, "size unknown")
        # Only the size-fitting zoom needs a size: near the edge it still widens.
        edge = BoundingBox(520, 90, 580, 190)
        self.assertEqual(unknown.update(0.1, 12.0, edge, FRAME, confidence=0.85), ("out", 4))
        self.assertEqual(unknown.last_reason, "target near the edge")

    def test_safety_zoom_out_is_bounded_and_ends_with_its_condition(self):
        controller = ZoomController(TABLE, lost_hfov=40.0, safety_step=1.5)
        for step in range(20):
            controller.update(step * 0.1, 20.0, box(), FRAME, confidence=0.85)
        edge = box(centre_x=560)
        self.assertEqual(controller.update(2.0, 20.0, edge, FRAME, confidence=0.85)[0], "out")
        self.assertEqual(controller.update(2.1, 26.0, edge, FRAME, confidence=0.85)[0], "out")
        # One stretch widens to 1.5 times where it started, then holds.
        self.assertEqual(controller.update(2.2, 30.0, edge, FRAME, confidence=0.85)[0], "stop")
        # Back from the edge, the zoom-out ends at once.
        self.assertEqual(controller.update(2.3, 24.0, box(), FRAME, confidence=0.85)[0], "stop")
        # Never past the search width, wherever it starts.
        wide = ZoomController(TABLE, lost_hfov=40.0)
        self.assertEqual(wide.update(0.0, 44.0, edge, FRAME, confidence=0.85)[0], "stop")
        self.assertEqual(wide.update(0.1, 38.0, edge, FRAME, confidence=0.85)[0], "out")

    def test_lost_widens_to_the_search_width_only_and_narrows_on_request(self):
        controller = ZoomController(TABLE, lost_hfov=40.0)
        self.assertEqual(controller.update(0.0, 25.0, None, FRAME, lost=True), ("out", 4))
        self.assertEqual(controller.update(0.1, 40.0, None, FRAME, lost=True), ("stop", 0))
        self.assertEqual(controller.update(0.2, 58.0, None, FRAME, lost=True), ("in", 2))
        self.assertEqual(controller.update(0.3, 40.0, None, FRAME, lost=True, narrow_to=20.0), ("in", 2))
        self.assertAlmostEqual(controller.last_target_hfov, 20.0)
        self.assertEqual(controller.update(0.4, 20.2, None, FRAME, lost=True, narrow_to=20.0), ("stop", 0))


class ZoomStateTest(unittest.TestCase):
    def test_estimate_integrates_commands_and_clamps(self):
        clock = [0.0]
        state = ZoomState(TABLE, 5543, {1: 640.0, 4: 1940.0}, now_fn=lambda: clock[0])
        state.commanded("in", 1)
        clock[0] = 1.0
        self.assertAlmostEqual(state.position(), 6183, delta=1)
        state.commanded("stop", 0)
        clock[0] = 2.0
        self.assertAlmostEqual(state.position(), 6183, delta=1)
        state.commanded("out", 4)
        clock[0] = 10.0
        self.assertEqual(state.position(), 0)
        self.assertEqual(state.source, "estimate")

    def test_estimate_of_an_absolute_move_stops_at_the_target(self):
        clock = [0.0]
        state = ZoomState(TABLE, 5543, {1: 640.0, 4: 1940.0}, now_fn=lambda: clock[0])
        state.commanded_to(4000, 4)
        clock[0] = 0.5
        self.assertAlmostEqual(state.position(), 4573, delta=1)
        clock[0] = 5.0
        self.assertEqual(state.position(), 4000)


class ViewScaleRecoveryTest(unittest.TestCase):
    def test_zoomed_in_box_recovers_only_when_scale_is_given(self):
        person = Detection("person", 0.9, BoundingBox(100, 20, 200, 220), track_id=7)
        marker = Detection("marker", 1.0, BoundingBox(140, 100, 160, 120))
        # Same person after a 2.4x zoom-in, new track id: 5.8 times the area.
        zoomed = Detection("person", 0.9, BoundingBox(30, 20, 270, 500), track_id=9)
        for scales, expected in (((None, None), "grace"), ((1.0, 2.4), "recovering")):
            tracker = MarkerTracker(confirmation_seconds=0, recovery_confirmation_seconds=0.12)
            tracker.update([person, marker], 10.0, view_scale=scales[0])
            self.assertEqual(tracker.update([zoomed], 10.1, view_scale=scales[1]).state, expected)


if __name__ == "__main__":
    unittest.main()
