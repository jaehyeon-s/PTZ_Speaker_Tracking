"""Zoom control from framing size, without a pose model.

The controller decides zoom in / out / stop from the target box, its
confidence and how fast the target moves. Sizes are kept as angles (degrees),
so a zoom change by itself does not look like the person changing size.
"""
from __future__ import annotations

import math
import socket
import threading
import time
from collections import deque

from .core import BoundingBox


class ZoomFovTable:
    """Zoom position <-> horizontal field of view, interpolated in log(FOV)."""

    def __init__(self, pairs: list[tuple[float, float]]) -> None:
        if len(pairs) < 2:
            raise ValueError("Zoom FOV table needs at least two position:hfov pairs")
        self._pairs = sorted((float(p), float(f)) for p, f in pairs)
        if any(f <= 0 for _, f in self._pairs):
            raise ValueError("Zoom FOV table values must be positive")
        self.min_position, self.max_position = self._pairs[0][0], self._pairs[-1][0]
        self.max_hfov = max(f for _, f in self._pairs)
        self.min_hfov = min(f for _, f in self._pairs)

    def hfov(self, position: float) -> float:
        position = min(self.max_position, max(self.min_position, position))
        for (p0, f0), (p1, f1) in zip(self._pairs, self._pairs[1:]):
            if position <= p1:
                share = (position - p0) / (p1 - p0) if p1 > p0 else 0.0
                return math.exp(math.log(f0) + share * (math.log(f1) - math.log(f0)))
        return self._pairs[-1][1]

    def position(self, hfov: float) -> float:
        hfov = min(self.max_hfov, max(self.min_hfov, hfov))
        for (p0, f0), (p1, f1) in zip(self._pairs, self._pairs[1:]):
            low, high = min(f0, f1), max(f0, f1)
            if low <= hfov <= high:
                share = (math.log(hfov) - math.log(f0)) / (math.log(f1) - math.log(f0)) if f0 != f1 else 0.0
                return p0 + share * (p1 - p0)
        return self._pairs[-1][0]


class ZoomController:
    """Chooses "in", "out" or "stop" each frame.

    Zoom in: only slowly, when the target is too small, slow and near the
    centre, and not within the cooldown after the last zoom movement.
    Zoom out: at once when the target nears an edge, moves fast for the current
    field of view, loses confidence, or is too large. The safety zoom-outs
    (edge, speed, confidence) need no size estimate, stop when their condition
    clears and never widen past `safety_hfov`.
    While the body is judged hidden the zoom holds still.
    """

    def __init__(
        self,
        table: ZoomFovTable,
        target_ratio: float = 0.55,
        tolerance: float = 0.15,
        upper_body_fraction: float = 0.5,
        delay_seconds: float = 0.4,
        response_seconds: float = 0.6,
        cooldown_seconds: float = 2.0,
        lost_hfov: float = 40.0,
        in_speed: int = 1,
        out_speed: int = 4,
        velocity_window_seconds: float = 2.5,
        centre_limit: tuple[float, float] = (0.15, 0.25),
        edge_limit: float = 0.35,
        ceiling_factor: float = 1.5,
        in_max_screen_speed: float = 0.10,
        search_marker_pixels: float = 14.0,
        search_speed: int = 2,
        max_marker_pixels: float = 45.0,
        upper_body_meters: float | None = None,
        marker_meters: float = 0.093,
        marker_min_pixels: float = 14.0,
        safety_margin: float = 1.3,
        safety_step: float = 1.5,
    ) -> None:
        if not 0 < target_ratio < 2 or not 0 < tolerance < 1 or not 0 < upper_body_fraction <= 1:
            raise ValueError("Zoom framing parameters are out of range")
        self._table = table
        self._target_ratio = target_ratio
        self._tolerance = tolerance
        self._upper_body_fraction = upper_body_fraction
        self._latency = delay_seconds + response_seconds
        self._cooldown_seconds = cooldown_seconds
        self._lost_hfov = lost_hfov
        self.lost_hfov = lost_hfov
        self._in_speed = in_speed
        self._out_speed = out_speed
        self._velocity_window = velocity_window_seconds
        self._centre_limit = centre_limit
        self._edge_limit = edge_limit
        self._ceiling_factor = ceiling_factor
        # Zoom-in needs the target to cross less than this share of the frame
        # width per second, whatever the current field of view is.
        self._in_max_screen_speed = in_max_screen_speed
        self._search_marker_pixels = search_marker_pixels
        # Zoom-in stops once the marker is this wide (640-wide frame); a guard
        # that does not depend on the person's box.
        self._max_marker_pixels = max_marker_pixels
        # With a known marker size the upper body's size in the picture follows
        # from the marker's: upper body pixels = marker pixels * (upper body m / marker m).
        self._upper_body_meters = upper_body_meters
        self._marker_meters = marker_meters
        self._marker_min_pixels = marker_min_pixels
        # Safety zoom-outs stop at this multiple of the field of view the speed
        # limit asks for (or at the size-based field of view, if that is wider).
        self._safety_margin = safety_margin
        # ... or at this multiple of the field of view the zoom-out started from.
        self._safety_step = safety_step
        self._safety_from: tuple[float, float] | None = None  # (hfov at start, last time the condition held)
        self.last_target_hfov: float | None = None
        self._marker_degrees: float | None = None
        self._marker_seen_at: float | None = None
        self._search_speed = search_speed
        self._speeds: deque[tuple[float, float]] = deque()
        self._person_degrees: float | None = None
        self._marker_factor: float | None = None
        self._confidence: float | None = None
        self._last_motion_at: float | None = None
        self._zooming_in = False
        self.last_reason = "idle"
        self.last_size_source = "none"
        self.last_min_hfov = 0.0
        self.last_ratio: float | None = None
        self.last_safety_hfov = 0.0

    def person_degrees(self) -> float | None:
        """Angular height of the whole person, as last measured or inferred."""
        return self._person_degrees

    def _measure(self, box, frame_shape, hfov, occluded, marker_box, now=0.0):
        height, width = frame_shape[:2]
        degrees_per_pixel = hfov / width
        if (
            self._upper_body_meters
            and marker_box is not None
            and (marker_box.x2 - marker_box.x1) >= self._marker_min_pixels
        ):
            upper_pixels = (marker_box.x2 - marker_box.x1) * self._upper_body_meters / self._marker_meters
            # Stored as the whole person's angular height, like the box-based estimate.
            self._marker_degrees = upper_pixels / self._upper_body_fraction * degrees_per_pixel
            self._marker_seen_at = now
        marker_degrees = (marker_box.x2 - marker_box.x1) * degrees_per_pixel if marker_box else None
        clipped = box.y2 >= height - 3 or box.y1 <= 2
        # Only a tall, complete box is a whole standing person. Zoomed in, the
        # detector often boxes just the head or the torso; taking that for the
        # whole person would make the target look small and zoom in further.
        whole = (box.y2 - box.y1) >= 2.0 * (box.x2 - box.x1)
        if not occluded and not clipped and whole:
            measured = (box.y2 - box.y1) * degrees_per_pixel
            if self._person_degrees is None:
                self._person_degrees = measured
            else:
                # A person's angular size changes with distance only, so slowly.
                step = max(-0.03, min(0.03, (measured - self._person_degrees) / self._person_degrees))
                self._person_degrees *= 1 + step
            if marker_degrees:
                factor = measured / marker_degrees
                self._marker_factor = factor if self._marker_factor is None else self._marker_factor + 0.1 * (factor - self._marker_factor)
        elif not occluded and marker_degrees and self._marker_factor:
            # The feet are out of the picture: the marker's size still tells the distance.
            inferred = marker_degrees * self._marker_factor
            if self._person_degrees is not None:
                self._person_degrees += 0.25 * (inferred - self._person_degrees)
            else:
                self._person_degrees = inferred

    def update(
        self,
        now: float,
        hfov: float,
        box: BoundingBox | None,
        frame_shape: tuple[int, ...],
        confidence: float | None = None,
        occluded: bool = False,
        target_velocity: float = 0.0,
        marker_box: BoundingBox | None = None,
        lost: bool = False,
        search_box: BoundingBox | None = None,
        narrow_to: float | None = None,
    ) -> tuple[str, int]:
        """`narrow_to`: while lost with nobody in view, narrow to this field of view."""
        self.last_target_hfov = None
        if lost and search_box is not None:
            action, speed, reason = self._search(hfov, search_box, frame_shape, marker_box)
        elif lost:
            action, speed, reason = self._lost(hfov, narrow_to)
        else:
            action, speed, reason = self._decide(
                now, hfov, box, frame_shape, confidence, occluded, target_velocity, marker_box
            )
        if action == "stop":
            self.last_target_hfov = None
        if action != "stop":
            self._last_motion_at = now
        self._zooming_in = action == "in"
        self.last_reason = reason
        return action, speed

    def _search(self, hfov, candidate, frame_shape, marker_box):
        """Second stage after a loss: a person is in view, so narrow onto that
        candidate until the marker is large enough to be confirmed."""
        height, width = frame_shape[:2]
        self._speeds.clear()
        marker_pixels = (marker_box.x2 - marker_box.x1) if marker_box is not None else 0.0
        if marker_pixels >= self._search_marker_pixels:
            return "stop", 0, "search: marker large enough"
        error_x = (candidate.center[0] - width / 2) / (width / 2)
        if abs(error_x) > 0.4:
            return "stop", 0, "search: centring candidate"
        clipped = candidate.y2 >= height - 3
        ratio = self._upper_body_fraction * (candidate.y2 - candidate.y1) / height
        if clipped or ratio >= self._target_ratio:
            return "stop", 0, "search: candidate fills the frame"
        if hfov <= self._table.min_hfov * 1.02:
            return "stop", 0, "search: zoom limit"
        self.last_target_hfov = max(self._table.min_hfov, hfov / 1.3)
        return "in", self._search_speed, "search: narrowing onto candidate"

    def _lost(self, hfov, narrow_to):
        """Lost with nobody in view: widen to the search width, never beyond;
        on request narrow again (the caller points at where the target was)."""
        self._speeds.clear()
        wide = min(self._lost_hfov, self._table.max_hfov)
        if narrow_to is not None:
            narrow_to = min(wide, max(self._table.min_hfov, narrow_to))
            self.last_target_hfov = narrow_to
            if hfov > narrow_to * 1.03:
                return "in", self._search_speed, "lost: narrowing to where it was last seen"
            return "stop", 0, "lost: narrowed"
        self.last_target_hfov = wide
        if hfov < wide * 0.98:
            return "out", self._out_speed, "lost: widening"
        if hfov > wide * 1.05:
            return "in", self._search_speed, "lost: back to the search width"
        return "stop", 0, "lost: wide"

    def _decide(self, now, hfov, box, frame_shape, confidence, occluded, target_velocity, marker_box):
        table = self._table
        if box is None or len(frame_shape) < 2:
            return "stop", 0, "no target"
        height, width = frame_shape[:2]
        self._measure(box, frame_shape, hfov, occluded, marker_box, now)
        # Target speed in degrees/second; errors are normalised by half the frame width.
        self._speeds.append((now, abs(target_velocity) * hfov / 2))
        while self._speeds and now - self._speeds[0][0] > self._velocity_window:
            self._speeds.popleft()
        ordered = sorted(value for _, value in self._speeds)
        fast = ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))]
        half_width_degrees = (box.x2 - box.x1) / 2 * hfov / width
        min_hfov = 2 * (self._ceiling_factor * fast * self._latency + half_width_degrees)
        # Same speed as a share of the frame width per second.
        screen_speed = fast / hfov
        self.last_min_hfov = min_hfov
        weaker = (
            confidence is not None
            and self._confidence is not None
            and (confidence < self._confidence - 0.2 or confidence < 0.35)
        )
        if confidence is not None and not occluded:
            self._confidence = confidence if self._confidence is None else self._confidence + 0.1 * (confidence - self._confidence)
        if occluded:
            return "stop", 0, "occluded: hold"
        # Size reference: the marker while it is seen large enough, otherwise the
        # height learned from complete boxes. With neither, only the size-fitting
        # zoom is withheld; the safety zoom-outs below still act.
        marker_fresh = self._marker_seen_at is not None and now - self._marker_seen_at <= 1.0
        size_degrees = self._marker_degrees if marker_fresh else self._person_degrees
        self.last_size_source = "marker" if marker_fresh else "box" if size_degrees is not None else "none"
        size_hfov = (
            None
            if size_degrees is None
            else self._upper_body_fraction * size_degrees * width / (height * self._target_ratio)
        )
        error_x = (box.center[0] - width / 2) / (width / 2)
        error_y = (box.y1 + 0.35 * (box.y2 - box.y1) - height / 2) / (height / 2)
        can_widen = hfov < table.max_hfov * 0.98
        if can_widen and hfov < min_hfov * 0.98:
            self.last_target_hfov = min(table.max_hfov, min_hfov)
            return "out", self._out_speed, "fast target for this field of view"
        # Safety zoom-outs: one stretch may widen to the largest of the field
        # of view the speed limit asks for (with a margin), the size-based one
        # and a fixed multiple of where it started; never past the search width.
        unsafe = abs(error_x) > self._edge_limit or weaker
        if unsafe:
            self._safety_from = (self._safety_from[0] if self._safety_from else hfov, now)
        elif self._safety_from is not None and now - self._safety_from[1] > 1.0:
            self._safety_from = None
        if unsafe:
            safety_hfov = min(
                table.max_hfov,
                self._lost_hfov,
                max(min_hfov * self._safety_margin, size_hfov or 0.0, self._safety_from[0] * self._safety_step),
            )
            self.last_safety_hfov = safety_hfov
            if hfov < safety_hfov * 0.98:
                self.last_target_hfov = safety_hfov
                reason = "target near the edge" if abs(error_x) > self._edge_limit else "confidence dropped"
                return "out", self._out_speed, reason
        # Independent of the size estimate: a box filling the frame, or a marker
        # far larger than at the wanted framing, means the picture is too tight.
        marker_pixels = (marker_box.x2 - marker_box.x1) if marker_box is not None else 0.0
        fills = (box.y1 <= 2 and box.y2 >= height - 3) or (box.x2 - box.x1) > 0.5 * width
        if can_widen and (fills or marker_pixels > self._max_marker_pixels * 1.3):
            self.last_target_hfov = min(table.max_hfov, hfov * 1.2)
            return "out", self._out_speed, "picture too tight"
        if size_degrees is None:
            return "stop", 0, "size unknown"
        ratio = self._upper_body_fraction * size_degrees * (width / hfov) / height
        self.last_ratio = ratio
        wanted = min(table.max_hfov, max(table.min_hfov, min_hfov * 1.1, size_hfov))
        self.last_target_hfov = wanted
        if can_widen and ratio > self._target_ratio * (1 + self._tolerance):
            return "out", self._out_speed, "target too large"
        tight = fills or marker_pixels > self._max_marker_pixels or box.y1 <= 2
        too_small = ratio < self._target_ratio * (1 - (self._tolerance / 2 if self._zooming_in else self._tolerance))
        centred = abs(error_x) <= self._centre_limit[0] and abs(error_y) <= self._centre_limit[1]
        rested = (
            self._zooming_in
            or self._last_motion_at is None
            or now - self._last_motion_at >= self._cooldown_seconds
        )
        slow = screen_speed <= self._in_max_screen_speed
        if tight:
            return "stop", 0, "picture tight: no zoom-in"
        if too_small and centred and rested and slow and hfov > wanted * 1.02 and hfov > table.min_hfov * 1.02:
            return "in", self._in_speed, "target small, slow and centred"
        return "stop", 0, "within range" if not too_small else "waiting to zoom in"


class ZoomState:
    """Current zoom position: read by VISCA inquiry when possible, else estimated.

    The estimate integrates the commanded zoom speed. It drifts, so a camera
    that answers the inquiry should have it enabled.
    """

    INQUIRY = bytes([0x81, 0x09, 0x04, 0x47, 0xFF])

    def __init__(
        self,
        table: ZoomFovTable,
        start_position: float,
        rates: dict[int, float],
        inquiry_host: str | None = None,
        inquiry_port: int = 0,
        poll_seconds: float = 0.25,
        now_fn=time.monotonic,
    ) -> None:
        self._table = table
        self._position = float(start_position)
        self._rates = dict(rates)
        self._now = now_fn
        self._moving: tuple[int, float] = (0, 0.0)
        self._moving_to: float | None = None
        self._updated_at = now_fn()
        self._lock = threading.Lock()
        self.source = "estimate"
        self._closed = threading.Event()
        self._thread = None
        if inquiry_host and inquiry_port:
            self._host, self._port, self._poll = inquiry_host, inquiry_port, poll_seconds
            self._thread = threading.Thread(target=self._poll_loop, daemon=True)
            self._thread.start()

    def _rate(self, speed: int) -> float:
        if not self._rates:
            return 0.0
        if speed in self._rates:
            return self._rates[speed]
        nearest = min(self._rates, key=lambda level: abs(level - speed))
        return self._rates[nearest] * speed / nearest

    def _advance(self, now: float) -> None:
        direction, rate = self._moving
        position = self._position + direction * rate * (now - self._updated_at)
        if self._moving_to is not None and (position - self._moving_to) * direction >= 0:
            position, self._moving, self._moving_to = self._moving_to, (0, 0.0), None
        self._position = min(self._table.max_position, max(self._table.min_position, position))
        self._updated_at = now

    def commanded(self, action: str, speed: int) -> None:
        """Tell the estimator which zoom command is now in effect."""
        with self._lock:
            now = self._now()
            self._advance(now)
            direction = 1 if action == "in" else -1 if action == "out" else 0
            self._moving = (direction, self._rate(speed) if direction else 0.0)
            self._moving_to = None

    def commanded_to(self, position: float, speed: int) -> None:
        """An absolute zoom to `position` was sent."""
        with self._lock:
            self._advance(self._now())
            direction = 1 if position > self._position else -1
            self._moving, self._moving_to = (direction, self._rate(speed)), float(position)

    def position(self) -> float:
        with self._lock:
            self._advance(self._now())
            return self._position

    def hfov(self) -> float:
        return self._table.hfov(self.position())

    def _poll_loop(self) -> None:
        failures = 0
        while not self._closed.is_set():
            value = self.inquire(self._host, self._port)
            if value is None:
                failures += 1
                if failures >= 5 and self.source == "estimate":
                    return  # never answered: stay on the estimate
            else:
                failures = 0
                with self._lock:
                    self._position, self._updated_at, self.source = float(value), self._now(), "inquiry"
            self._closed.wait(self._poll)

    @classmethod
    def inquire(cls, host: str, port: int, timeout: float = 0.5) -> int | None:
        try:
            with socket.create_connection((host, port), timeout=timeout) as connection:
                connection.sendall(cls.INQUIRY)
                connection.settimeout(timeout)
                data = b""
                while not data.endswith(b"\xff") and len(data) < 32:
                    chunk = connection.recv(32)
                    if not chunk:
                        break
                    data += chunk
        except OSError:
            return None
        start = data.find(b"\x90\x50")
        if start < 0 or len(data) < start + 7:
            return None
        value = 0
        for byte in data[start + 2:start + 6]:
            value = (value << 4) | (byte & 0x0F)
        return value

    def close(self) -> None:
        self._closed.set()
