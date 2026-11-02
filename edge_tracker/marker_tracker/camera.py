from __future__ import annotations

import logging
import base64
import socket
import struct
import threading
import re
from urllib.parse import urlencode
from urllib.request import (
    HTTPBasicAuthHandler,
    HTTPDigestAuthHandler,
    HTTPPasswordMgrWithDefaultRealm,
    Request,
    build_opener,
    urlopen,
)
from collections.abc import Sequence
from typing import Protocol

from .core import BoundingBox


class CameraTrackingAdapter(Protocol):
    def update(
        self,
        target: BoundingBox | None,
        frame_shape: tuple[int, ...],
        speed_limit: int | None = None,
        immediate_stop: bool = True,
        allow_coast: bool = False,
    ) -> None: ...

    def handoff(self, target: BoundingBox) -> None: ...

    def release(self) -> None: ...

    def status(self) -> str: ...

    def close(self) -> None: ...


class LoggingCameraAdapter:
    """Temporary adapter until the Qon camera control API is available."""

    def update(
        self,
        target: BoundingBox | None,
        frame_shape: tuple[int, ...],
        speed_limit: int | None = None,
        immediate_stop: bool = True,
        allow_coast: bool = False,
    ) -> None:
        del target, frame_shape, speed_limit, immediate_stop, allow_coast

    def handoff(self, target: BoundingBox) -> None:
        logging.info("PTZ handoff requested: person_box=%s", target)

    def release(self) -> None:
        logging.info("PTZ tracking release requested: restore camera default")

    def status(self) -> str:
        return "log"

    def close(self) -> None:
        pass


class ViscaOverIpCameraAdapter:
    """VISCA over IP pan/tilt adapter using continuous drive commands."""

    _STOP = 0x03
    _LEFT = 0x01
    _RIGHT = 0x02
    _UP = 0x01
    _DOWN = 0x02

    def __init__(
        self,
        host: str,
        port: int = 52381,
        packet_mode: str = "sony-ip",
        pan_speed: int = 8,
        tilt_speed: int = 6,
        deadzone_x: float = 0.12,
        deadzone_y: float = 0.12,
        invert_pan: bool = False,
        invert_tilt: bool = False,
        min_interval_seconds: float = 0.12,
        timeout_seconds: float = 0.5,
        log_commands: bool = False,
        now_fn=None,
        socket_factory=None,
    ) -> None:
        if not host:
            raise ValueError("VISCA host is required")
        if packet_mode not in {"sony-ip", "raw"}:
            raise ValueError("VISCA packet_mode must be 'sony-ip' or 'raw'")
        if not 1 <= pan_speed <= 24 or not 1 <= tilt_speed <= 20:
            raise ValueError("VISCA pan_speed must be 1..24 and tilt_speed must be 1..20")
        if deadzone_x < 0 or deadzone_y < 0 or min_interval_seconds < 0:
            raise ValueError("VISCA deadzone and interval values must be non-negative")
        self._address = (host, port)
        self._packet_mode = packet_mode
        self._pan_speed = pan_speed
        self._tilt_speed = tilt_speed
        self._deadzone_x = deadzone_x
        self._deadzone_y = deadzone_y
        self._invert_pan = invert_pan
        self._invert_tilt = invert_tilt
        self._min_interval_seconds = min_interval_seconds
        self._log_commands = log_commands
        self._now_fn = now_fn
        self._socket_factory = socket_factory or socket.socket
        self._sequence = 1
        self._last_sent_at: float | None = None
        self._last_drive: tuple[int, int, int, int] | None = None
        self._last_status = "stop"
        self._socket = self._socket_factory(socket.AF_INET, socket.SOCK_DGRAM)
        self._socket.settimeout(timeout_seconds)
        logging.info(
            "VISCA camera control enabled: host=%s port=%s mode=%s pan_speed=%s tilt_speed=%s",
            host,
            port,
            packet_mode,
            pan_speed,
            tilt_speed,
        )

    def update(
        self,
        target: BoundingBox | None,
        frame_shape: tuple[int, ...],
        speed_limit: int | None = None,
        immediate_stop: bool = True,
        allow_coast: bool = False,
    ) -> None:
        del speed_limit, immediate_stop, allow_coast
        if target is None:
            self._drive(self._STOP, self._STOP, force=False)
            return
        if len(frame_shape) < 2:
            return
        height, width = frame_shape[:2]
        if width <= 0 or height <= 0:
            return
        target_x, target_y = target.center
        error_x = (target_x - (width / 2)) / (width / 2)
        error_y = (target_y - (height / 2)) / (height / 2)
        pan_direction = self._axis_direction(
            error_x,
            self._deadzone_x,
            negative_direction=self._LEFT,
            positive_direction=self._RIGHT,
            invert=self._invert_pan,
        )
        tilt_direction = self._axis_direction(
            error_y,
            self._deadzone_y,
            negative_direction=self._UP,
            positive_direction=self._DOWN,
            invert=self._invert_tilt,
        )
        self._drive(pan_direction, tilt_direction, force=False)

    def handoff(self, target: BoundingBox) -> None:
        logging.info("PTZ VISCA tracking target selected: person_box=%s", target)

    def release(self) -> None:
        logging.info("PTZ VISCA tracking release requested: stopping pan/tilt")
        self._drive(self._STOP, self._STOP, force=True)

    def status(self) -> str:
        return self._last_status

    def close(self) -> None:
        self._drive(self._STOP, self._STOP, force=True)
        self._socket.close()

    def _drive(self, pan_direction: int, tilt_direction: int, force: bool) -> None:
        drive = (self._pan_speed, self._tilt_speed, pan_direction, tilt_direction)
        now = self._now()
        if (
            not force
            and drive == self._last_drive
            and self._last_sent_at is not None
            and now - self._last_sent_at < self._min_interval_seconds
        ):
            return
        self._send(
            bytes(
                [
                    0x81,
                    0x01,
                    0x06,
                    0x01,
                    self._pan_speed,
                    self._tilt_speed,
                    pan_direction,
                    tilt_direction,
                    0xFF,
                ]
            )
        )
        self._last_drive = drive
        self._last_status = self._format_drive_status(pan_direction, tilt_direction)
        self._last_sent_at = now

    def _send(self, payload: bytes) -> None:
        packet = payload if self._packet_mode == "raw" else self._sony_ip_packet(payload)
        if self._log_commands:
            logging.info("VISCA send %s", packet.hex(" "))
        try:
            self._socket.sendto(packet, self._address)
        except OSError as exc:
            logging.warning("Failed to send VISCA command to %s: %s", self._address, exc)

    def _sony_ip_packet(self, payload: bytes) -> bytes:
        sequence = self._sequence
        self._sequence = (self._sequence + 1) & 0xFFFFFFFF
        return struct.pack(">HHI", 0x0100, len(payload), sequence) + payload

    def _format_drive_status(self, pan_direction: int, tilt_direction: int) -> str:
        pan = {
            self._LEFT: "left",
            self._RIGHT: "right",
            self._STOP: "stop",
        }[pan_direction]
        tilt = {
            self._UP: "up",
            self._DOWN: "down",
            self._STOP: "stop",
        }[tilt_direction]
        if pan == "stop" and tilt == "stop":
            return "stop"
        if pan == "stop":
            return f"tilt:{tilt}"
        if tilt == "stop":
            return f"pan:{pan}"
        return f"pan:{pan} tilt:{tilt}"

    def _now(self) -> float:
        if self._now_fn is not None:
            return self._now_fn()
        import time

        return time.monotonic()

    @staticmethod
    def _axis_direction(
        error: float,
        deadzone: float,
        negative_direction: int,
        positive_direction: int,
        invert: bool,
    ) -> int:
        if abs(error) <= deadzone:
            return ViscaOverIpCameraAdapter._STOP
        direction = positive_direction if error > 0 else negative_direction
        if not invert:
            return direction
        return negative_direction if direction == positive_direction else positive_direction


class QonHttpCameraAdapter:
    """Qon PTZ adapter using the observed HTTP ptzctrl.cgi command surface."""

    _TRACK_CONFIG_PATH = "/data/track.conf"
    _TRACK_REFRESH_VISCA = (0x81, 0x0A, 0x01, 0x04, 0x1D, 0x17, 0xFF)
    # [room_01 수정] 지연 보상으로 깎을 수 있는 오차의 최대 비율.
    # 계속 걷는 대상은 카메라가 움직여도 가까워지지 않으므로 절반까지만 깎음.
    _MAX_COMPENSATED_FRACTION = 0.5

    def __init__(
        self,
        camera_url: str,
        username: str | None = None,
        password: str | None = None,
        min_speed: int = 2,
        max_speed: int = 10,
        deadzone: float = 0.12,
        stop_deadzone: float = 0.08,
        stop_debounce_seconds: float = 0.12,
        lost_target_hold_seconds: float = 0.20,
        command_min_interval_seconds: float = 0.12,
        command_ttl_seconds: float = 0.35,
        timeout_seconds: float = 1.0,
        configure_supervisor_actuator: bool = True,
        auth_mode: str = "basic",
        async_commands: bool = True,
        log_commands: bool = False,
        close_join_timeout_seconds: float = 1.0,
        # [room_01 수정] 신규 인자 6개. 기본값은 수정 전 동작과 같음.
        full_speed_error: float = 1.0,
        uniform_error_scale: bool = False,
        axis_switch_ratio: float = 1.0,
        response_delay_seconds: float = 0.0,
        pan_rate_per_speed: float = 0.075,
        tilt_rate_per_speed: float = 0.05,
        now_fn=None,
        opener=None,
        speed_map: Sequence[tuple[float, int]] | None = None,
        predict_target_seconds: float = 0.0,
        stop_lookahead_seconds: float = 0.0,
        level_speeds: dict[int, float] | None = None,
        tilt_level_speeds: dict[int, float] | None = None,
        axis_mode: str = "dominant",
        axis_alternate_seconds: float = 0.35,
        lost_coast_seconds: float = 0.0,
        lost_coast_speed: int = 2,
        zoom_max_burst_seconds: float = 0.0,
        zoom_burst_rest_seconds: float = 0.3,
        tilt_deadzone: float | None = None,
        tilt_stop_deadzone: float | None = None,
        predict_axes: str = "both",
        axis_reverse_hold_seconds: float = 0.0,
    ) -> None:
        if not camera_url:
            raise ValueError("Qon camera URL is required")
        if predict_target_seconds < 0 or stop_lookahead_seconds < 0:
            raise ValueError("Qon prediction and look-ahead times must not be negative")
        if (predict_target_seconds or stop_lookahead_seconds) and not level_speeds:
            raise ValueError("Qon prediction and stop look-ahead need level_speeds")
        if axis_mode not in {"dominant", "diagonal", "alternate"}:
            raise ValueError("Qon axis_mode must be one of dominant, diagonal, alternate")
        if axis_alternate_seconds <= 0 or lost_coast_seconds < 0 or lost_coast_speed < 1:
            raise ValueError("Qon axis alternation and coast settings are invalid")
        if predict_axes not in {"both", "pan"} or axis_reverse_hold_seconds < 0:
            raise ValueError("Qon predict_axes must be both or pan and the reverse hold non-negative")
        if min_speed < 1 or max_speed < 1:
            raise ValueError("Qon PTZ speeds must be positive")
        if (
            deadzone < 0
            or stop_deadzone < 0
            or stop_deadzone > deadzone
            or stop_debounce_seconds < 0
            or lost_target_hold_seconds < 0
            or command_min_interval_seconds < 0
            or command_ttl_seconds < 0
        ):
            raise ValueError("Qon PTZ thresholds and timing values are invalid")
        if (
            # [room_01 수정] 신규 인자 범위 검사.
            not deadzone < full_speed_error <= 1.0
            or axis_switch_ratio < 1.0
            or response_delay_seconds < 0
            or pan_rate_per_speed < 0
            or tilt_rate_per_speed < 0
        ):
            raise ValueError("Qon PTZ response shaping values are invalid")
        if auth_mode not in {"none", "basic", "digest"}:
            raise ValueError("Qon auth_mode must be one of none, basic, digest")
        if "://" not in camera_url:
            camera_url = f"http://{camera_url}"
        self._camera_url = camera_url.rstrip("/")
        self._username = username
        self._password = password
        self._auth_mode = auth_mode
        self._min_speed = min_speed
        self._max_speed = max(min_speed, max_speed)
        self._deadzone = deadzone
        self._stop_deadzone = stop_deadzone
        self._stop_debounce_seconds = stop_debounce_seconds
        self._lost_target_hold_seconds = lost_target_hold_seconds
        self._command_min_interval_seconds = command_min_interval_seconds
        self._command_ttl_seconds = command_ttl_seconds
        self._timeout_seconds = timeout_seconds
        self._configure_supervisor_actuator = configure_supervisor_actuator
        self._async_commands = async_commands
        self._log_commands = log_commands
        self._close_join_timeout_seconds = close_join_timeout_seconds
        self._full_speed_error = full_speed_error
        self._uniform_error_scale = uniform_error_scale
        self._axis_switch_ratio = axis_switch_ratio
        # [room_01 수정] 응답 지연 보상용 상태.
        # room_01에서 ptzstop을 보낸 뒤에도 화면이 약 0.4~0.5초 더 움직였음.
        # 측정 오차만 보고 멈추면 이미 목표를 지나쳐 반대 방향 명령이 나가고,
        # 이것이 위아래/좌우 떨림(헌팅)으로 보였음.
        # The picture answers a command only after the camera and the video
        # stream delay, so the measured error is already out of date while
        # the camera moves. Steering on the error minus the motion still in
        # the pipeline (speed x rate x delay) brakes before the target instead
        # of past it. Rates are in half frame widths per second per speed step.
        self._response_delay_seconds = response_delay_seconds
        self._pan_rate_per_speed = pan_rate_per_speed
        self._tilt_rate_per_speed = tilt_rate_per_speed
        self._moving_since: float | None = None
        self._now_fn = now_fn
        self._opener = opener or self._build_opener()
        # Optional response tuning; all off by default (see _compensated_error).
        self._speed_map = sorted(speed_map) if speed_map else None
        self._predict_target_seconds = predict_target_seconds
        self._stop_lookahead_seconds = stop_lookahead_seconds
        self._level_speeds = dict(level_speeds or {})
        self._tilt_level_speeds = dict(tilt_level_speeds or level_speeds or {})
        # dominant: one axis per command (original behaviour). diagonal: both
        # axes in one command (leftup...). alternate: both axes, taking turns.
        self._axis_mode = axis_mode
        self._axis_alternate_seconds = axis_alternate_seconds
        self._pan_active = False
        self._tilt_active = False
        # Two-axis modes only: the tilt axis may use its own, wider deadzone,
        # and an axis may be made to wait before it reverses.
        self._tilt_deadzone = deadzone if tilt_deadzone is None else tilt_deadzone
        self._tilt_stop_deadzone = stop_deadzone if tilt_stop_deadzone is None else tilt_stop_deadzone
        self._predict_axes = predict_axes
        self._axis_reverse_hold_seconds = axis_reverse_hold_seconds
        self._axis_last_moved: dict[str, tuple[str, float]] = {}
        # Image speed of every level relative to the zoom at which the level
        # speeds were measured (1.0 = same zoom); set by the zoom control.
        self._view_gain = 1.0
        self._pending_zoom: tuple[str, int] | None = None
        self._last_zoom: tuple[str, int] = ("stop", 0)
        self._last_zoom_sent_at = 0.0
        # A continuous zoom runs until zoomstop arrives. With a burst limit the
        # adapter stops it by itself after this long, also when nobody asks.
        self._zoom_max_burst_seconds = zoom_max_burst_seconds
        self._zoom_burst_rest_seconds = zoom_burst_rest_seconds
        self._zoom_burst_started_at: float | None = None
        self._zoom_rest_until = 0.0
        self._zoom_stop_failed = False
        # After the target disappears, keep panning the way it was going.
        self._lost_coast_seconds = lost_coast_seconds
        self._lost_coast_speed = lost_coast_speed
        self._coast_direction: str | None = None
        self._last_pan_direction: tuple[str, float] | None = None
        self._motion_history: list[tuple[float, float, float]] = []
        self._lost_at: float | None = None
        self._last_observation: tuple[float, float, float, float, float] | None = None
        self._previous_error: tuple[float, float, float] | None = None
        self._target_velocity = (0.0, 0.0)
        self._last_command = "ptzstop"
        self._last_speed = 0
        self._last_sent_at = 0.0
        self._deadzone_since: float | None = None
        self._target_missing_since: float | None = None
        self.last_error: Exception | None = None
        self._pending_command: tuple[str, int, int] | None = None
        self._closed = False
        self._condition = threading.Condition()
        self._worker: threading.Thread | None = None
        if self._async_commands:
            self._worker = threading.Thread(target=self._send_loop, daemon=True)
            self._worker.start()
        logging.info(
            "Qon HTTP camera control enabled: url=%s min_speed=%s max_speed=%s "
            "deadzone=%s stop_deadzone=%s stop_debounce=%.2f lost_hold=%.2f "
            "async=%s auth=%s log_commands=%s",
            self._camera_url,
            self._min_speed,
            self._max_speed,
            self._deadzone,
            self._stop_deadzone,
            self._stop_debounce_seconds,
            self._lost_target_hold_seconds,
            self._async_commands,
            self._auth_mode,
            self._log_commands,
        )
        if self._configure_supervisor_actuator:
            self._apply_supervisor_actuator_mode()

    def update(
        self,
        target: BoundingBox | None,
        frame_shape: tuple[int, ...],
        speed_limit: int | None = None,
        immediate_stop: bool = True,
        allow_coast: bool = False,
    ) -> None:
        now = self._now()
        if target is None:
            self._previous_error = None
            self._target_velocity = (0.0, 0.0)
            self._deadzone_since = None
            self._pan_active = self._tilt_active = False
            if self._coast(now, allow_coast):
                return
            if immediate_stop or self._last_command == "ptzstop":
                self._target_missing_since = None
                self._stop()
                return
            if self._target_missing_since is None:
                self._target_missing_since = now
            if now - self._target_missing_since >= self._lost_target_hold_seconds:
                self._stop()
            return
        self._target_missing_since = None
        self._coast_direction = None
        self._lost_at = None
        if self._axis_mode == "dominant":
            command, speed = self._command_for_target(target, frame_shape, speed_limit, now)
            speeds = (speed, speed)
        else:
            command, speeds = self._two_axis_command(target, frame_shape, speed_limit, now)
            speed = speeds if command in self._DIAGONALS else speeds[0]
        self._remember_observation(target, frame_shape, now)
        if command == "ptzstop":
            if self._last_command == "ptzstop" or self._stop_debounce_seconds == 0:
                self._deadzone_since = None
                self._stop()
                return
            if self._deadzone_since is None:
                self._deadzone_since = now
            if now - self._deadzone_since >= self._stop_debounce_seconds:
                self._deadzone_since = None
                self._stop()
            return
        self._deadzone_since = None
        # [room_01 수정] 방향이 바뀐 시각 기록(지연 보상에 사용).
        if command != self._last_command:
            self._moving_since = now
        if (
            command != self._last_command
            or (
                speed != self._last_speed
                and now - self._last_sent_at >= self._command_min_interval_seconds
            )
            or now - self._last_sent_at >= self._command_ttl_seconds
        ):
            self._send(command, speeds[0], speeds[1])
            self._last_command = command
            self._last_speed = speed
            self._last_sent_at = now
        for pan in ("left", "right"):
            if command.startswith(pan):
                self._last_pan_direction = (pan, now)

    def _remember_observation(self, target: BoundingBox, frame_shape: tuple[int, ...], now: float) -> None:
        height, width = frame_shape[:2]
        if width <= 0 or height <= 0:
            return
        centre_x, centre_y = target.center
        self._last_observation = (
            now,
            (centre_x - width / 2) / (width / 2),
            (centre_y - height / 2) / (height / 2),
            self._target_velocity[0],
            self._target_velocity[1],
        )

    def predicted_error(self, now: float | None = None) -> tuple[float, float] | None:
        """Where the target last seen should be now, as a normalised error.

        The last observation is carried forward with the target's estimated
        speed and the camera motion commanded since. None if nothing was seen.
        """
        if self._last_observation is None:
            return None
        now = self._now() if now is None else now
        seen_at, error_x, error_y, velocity_x, velocity_y = self._last_observation
        delay = self._predict_target_seconds
        moved_x, moved_y = self._camera_motion(seen_at - delay, now)
        elapsed = now - seen_at + delay
        return error_x + velocity_x * elapsed - moved_x, error_y + velocity_y * elapsed - moved_y

    def follow_prediction(self, speed_limit: int | None = None) -> bool:
        """Steer to the predicted position while the target is not detected."""
        now = self._now()
        predicted = self.predicted_error(now)
        if predicted is None:
            return False
        error_x = predicted[0]
        direction = "right" if error_x > 0 else "left"
        threshold = self._stop_deadzone if self._last_command == direction else self._deadzone
        if abs(error_x) <= threshold:
            self._stop()
            return True
        speed = self._speed_for(abs(error_x), speed_limit)
        if (
            direction != self._last_command
            or speed != self._last_speed
            or now - self._last_sent_at >= self._command_ttl_seconds
        ):
            self._send(direction, speed, speed)
            self._last_command = direction
            self._last_speed = speed
            self._last_sent_at = now
        return True

    def _coast(self, now: float, allowed: bool) -> bool:
        """Keep panning in the last direction for a while after the target is lost."""
        if not self._lost_coast_seconds:
            return False
        if self._lost_at is None:
            self._lost_at = now
            recent = self._last_pan_direction
            self._coast_direction = recent[0] if recent and now - recent[1] <= 1.0 else None
        if not allowed or self._coast_direction is None or now - self._lost_at > self._lost_coast_seconds:
            return False
        speed = self._lost_coast_speed
        if self._last_command != self._coast_direction or self._last_speed != speed or (
            now - self._last_sent_at >= self._command_ttl_seconds
        ):
            self._send(self._coast_direction, speed, speed)
            self._last_command = self._coast_direction
            self._last_speed = speed
            self._last_sent_at = now
        return True

    def handoff(self, target: BoundingBox) -> None:
        logging.info("PTZ Qon HTTP tracking target selected: person_box=%s", target)

    def release(self) -> None:
        logging.info("PTZ Qon HTTP tracking release requested: stopping pan/tilt")
        if self._coast_direction is None or self._lost_at is None:
            self._stop()

    def status(self) -> str:
        if self._last_command == "ptzstop":
            return "ptzstop"
        if isinstance(self._last_speed, tuple):
            return f"{self._last_command}:{self._last_speed[0]}/{self._last_speed[1]}"
        return f"{self._last_command}:{self._last_speed}"

    def close(self) -> None:
        self._stop()
        if self._worker is not None:
            with self._condition:
                self._closed = True
                self._condition.notify()
            self._worker.join(timeout=self._close_join_timeout_seconds)
        self._send_with_warning("ptzstop", 10, 10)
        if self._last_zoom[0] != "stop" or self._zoom_stop_failed:
            self._send_with_warning("zoomstop", 0, None)

    def _command_for_target(
        self,
        target: BoundingBox | None,
        frame_shape: tuple[int, ...],
        speed_limit: int | None = None,
        now: float | None = None,
    ) -> tuple[str, int]:
        if target is None or len(frame_shape) < 2:
            return "ptzstop", 0
        height, width = frame_shape[:2]
        if width <= 0 or height <= 0:
            return "ptzstop", 0
        # [room_01 수정] 오차 계산을 _normalized_error로 분리하고, 진행 중인 명령이
        # 앞으로 줄여 줄 오차를 미리 뺌(_compensated_error).
        error_x, error_y = self._normalized_error(target, width, height, now)
        error_x = self._compensated_error(
            error_x, "right", "left", self._pan_rate_per_speed
        )
        error_y = self._compensated_error(
            error_y, "down", "up", self._tilt_rate_per_speed
        )
        horizontal = abs(error_x) >= abs(error_y)
        # [room_01 수정] 축 전환 hysteresis 추가.
        # 기존: 매 프레임 더 큰 오차 축 선택 -> right/up/right/up 이 0.1초 간격으로 반복됨.
        # Only one axis moves per command. Without hysteresis two similar
        # errors make the command flip between pan and tilt every few frames.
        if self._last_command in ("left", "right"):
            if abs(error_x) > self._stop_deadzone:
                horizontal = abs(error_y) < self._axis_switch_ratio * abs(error_x)
        elif self._last_command in ("up", "down"):
            if abs(error_y) > self._stop_deadzone:
                horizontal = abs(error_x) >= self._axis_switch_ratio * abs(error_y)
        if horizontal:
            direction = "right" if error_x > 0 else "left"
            magnitude = abs(error_x)
        else:
            direction = "down" if error_y > 0 else "up"
            magnitude = abs(error_y)
        threshold = (
            self._stop_deadzone
            if self._last_command == direction
            else self._deadzone
        )
        if magnitude <= threshold:
            return "ptzstop", 0
        return direction, self._speed_for(magnitude, speed_limit)

    _DIAGONALS = ("leftup", "leftdown", "rightup", "rightdown")

    def set_view_gain(self, gain: float) -> None:
        """Zoomed in by `gain`, each level moves the picture `gain` times faster."""
        self._view_gain = max(0.1, gain)

    def _speed_for(self, magnitude: float, speed_limit: int | None) -> int:
        # Choose the level as if the error were measured at the reference zoom.
        magnitude = magnitude / self._view_gain
        speed_span = self._max_speed - self._min_speed
        scaled = min(
            1.0,
            max(
                0.0,
                (magnitude - self._deadzone)
                # [room_01 수정] 속도 곡선 분모를 (1 - deadzone)에서 (full_speed_error - deadzone)으로.
                # 같은 오차에서 더 빠른 속도가 나옴.
                / max(self._full_speed_error - self._deadzone, 1e-6),
            ),
        )
        speed = int(round(self._min_speed + scaled * speed_span))
        if self._speed_map:
            speed = self._min_speed
            for error_threshold, level in self._speed_map:
                if magnitude >= error_threshold:
                    speed = level
            speed = max(self._min_speed, min(self._max_speed, speed))
        if speed_limit is not None:
            speed = min(speed, max(self._min_speed, speed_limit))
        return speed

    def _two_axis_command(
        self,
        target: BoundingBox,
        frame_shape: tuple[int, ...],
        speed_limit: int | None,
        now: float,
    ) -> tuple[str, tuple[int, int]]:
        """Pan and tilt decided independently, each with its own deadzone hysteresis."""
        if len(frame_shape) < 2 or frame_shape[0] <= 0 or frame_shape[1] <= 0:
            return "ptzstop", (0, 0)
        height, width = frame_shape[:2]
        error_x, error_y = self._normalized_error(target, width, height, now)
        self._pan_active = abs(error_x) > (self._stop_deadzone if self._pan_active else self._deadzone)
        self._tilt_active = abs(error_y) > (
            self._tilt_stop_deadzone if self._tilt_active else self._tilt_deadzone
        )
        pan = ("right" if error_x > 0 else "left") if self._pan_active else ""
        tilt = ("down" if error_y > 0 else "up") if self._tilt_active else ""
        pan = self._held_before_reversing("pan", pan, now)
        tilt = self._held_before_reversing("tilt", tilt, now)
        pan_speed = self._speed_for(abs(error_x), speed_limit) if pan else 0
        tilt_speed = self._speed_for(abs(error_y), speed_limit) if tilt else 0
        if not pan and not tilt:
            return "ptzstop", (0, 0)
        if pan and tilt:
            if self._axis_mode == "diagonal":
                return pan + tilt, (pan_speed, tilt_speed)
            # Camera without diagonal commands: the two axes take turns.
            if int(now / self._axis_alternate_seconds) % 2 == 0:
                return pan, (pan_speed, pan_speed)
            return tilt, (tilt_speed, tilt_speed)
        if pan:
            return pan, (pan_speed, pan_speed)
        return tilt, (tilt_speed, tilt_speed)

    def _held_before_reversing(self, axis: str, direction: str, now: float) -> str:
        """An axis that just moved one way waits before it may move the other way."""
        if not direction:
            return direction
        last = self._axis_last_moved.get(axis)
        if (
            last is not None
            and last[0] != direction
            and now - last[1] < self._axis_reverse_hold_seconds
        ):
            return ""
        self._axis_last_moved[axis] = (direction, now)
        return direction

    def target_velocity(self) -> float:
        """Horizontal target speed in half frame widths per second (0 if not estimated)."""
        return self._target_velocity[0]

    def zoom(self, action: str, speed: int) -> str:
        """Continuous zoom: "in", "out" or "stop". Re-sent while moving, like pan and tilt.

        Returns the action now in effect, which is "stop" while a burst limit
        holds the zoom still.
        """
        now = self._now()
        if action != "stop" and self._zoom_max_burst_seconds > 0:
            if now < self._zoom_rest_until:
                action = "stop"
            elif (
                self._zoom_burst_started_at is not None
                and self._last_zoom[0] != "stop"
                and now - self._zoom_burst_started_at >= self._zoom_max_burst_seconds
            ):
                self._zoom_rest_until = now + self._zoom_burst_rest_seconds
                action = "stop"
        wanted = (action, speed if action != "stop" else 0)
        if wanted == self._last_zoom and (
            (action == "stop" and not self._zoom_stop_failed)
            or (action != "stop" and now - self._last_zoom_sent_at < self._command_ttl_seconds)
        ):
            return action
        if action == "stop":
            self._zoom_burst_started_at = None
        elif self._last_zoom[0] != action or self._zoom_burst_started_at is None:
            self._zoom_burst_started_at = now
        command = {"in": "zoomin", "out": "zoomout", "stop": "zoomstop"}[action]
        self._last_zoom, self._last_zoom_sent_at = wanted, now
        self._queue_zoom(command, wanted[1], None)
        return action

    def zoom_to(self, position: int, speed: int) -> None:
        """Absolute zoom to a position (PTZOptics-style zoomto, four hex digits)."""
        self._last_zoom, self._last_zoom_sent_at = ("to", int(position)), self._now()
        self._zoom_burst_started_at = None
        self._queue_zoom("zoomto", speed, f"{max(0, int(position)):04X}")

    def _queue_zoom(self, command: str, speed: int, argument: str | None) -> None:
        if not self._async_commands:
            self._send_zoom(command, speed, argument)
            return
        with self._condition:
            self._pending_zoom = (command, speed, argument)
            self._condition.notify()

    def _send_zoom(self, command: str, speed: int, argument: str | None) -> None:
        sent = self._send_with_warning(command, speed, argument)
        if command == "zoomstop":
            # A stop that did not get through is sent again at the next request.
            self._zoom_stop_failed = not sent

    def _zoom_deadline(self) -> float | None:
        if (
            self._zoom_max_burst_seconds <= 0
            or self._zoom_burst_started_at is None
            or self._last_zoom[0] not in {"in", "out"}
        ):
            return None
        return self._zoom_burst_started_at + self._zoom_max_burst_seconds

    def last_seen_error(self, now: float | None = None) -> tuple[float, float] | None:
        """Where the place the target was last seen at lies now (no target motion assumed)."""
        if self._last_observation is None:
            return None
        now = self._now() if now is None else now
        seen_at, error_x, error_y, _, _ = self._last_observation
        moved_x, moved_y = self._camera_motion(seen_at - self._predict_target_seconds, now)
        return error_x - moved_x, error_y - moved_y

    def search_pan(self, direction: str | None, speed: int) -> None:
        """Pan at a fixed level while searching for a lost target; None stops."""
        now = self._now()
        if direction not in {"left", "right"}:
            self._stop()
            return
        if (
            direction != self._last_command
            or speed != self._last_speed
            or now - self._last_sent_at >= self._command_ttl_seconds
        ):
            self._send(direction, speed, speed)
            self._last_command = direction
            self._last_speed = speed
            self._last_sent_at = now

    def _level_velocity(self, command: str, speed: int, speed_y: int | None = None) -> tuple[float, float]:
        """Rate at which a command reduces the normalised error, per second.

        Level speeds are image speeds in pixels/second on a 640x360 frame;
        errors are normalised by half the frame size.
        """
        if command in self._DIAGONALS:
            pan = "left" if command.startswith("left") else "right"
            tilt = "up" if command.endswith("up") else "down"
            return (
                self._level_velocity(pan, speed)[0],
                self._level_velocity(tilt, speed if speed_y is None else speed_y)[1],
            )
        table = self._level_speeds if command in ("left", "right") else self._tilt_level_speeds
        if command == "ptzstop" or not table:
            return 0.0, 0.0
        levels = sorted(table)
        if speed in table:
            pixels = table[speed]
        elif speed < levels[0]:
            pixels = table[levels[0]] * speed / levels[0]
        else:
            # Interpolate between measured levels; extrapolate from the top two.
            upper = next((level for level in levels if level > speed), levels[-1])
            lower = max((level for level in levels if level < upper), default=levels[0])
            if upper == lower:
                pixels = table[lower]
            else:
                slope = (table[upper] - table[lower]) / (upper - lower)
                pixels = table[lower] + slope * (speed - lower)
        pixels *= self._view_gain
        if command == "right":
            return pixels / 320.0, 0.0
        if command == "left":
            return -pixels / 320.0, 0.0
        if command == "down":
            return 0.0, pixels / 180.0
        return 0.0, -pixels / 180.0

    def _camera_motion(self, start: float, end: float) -> tuple[float, float]:
        """Commanded camera motion between two times, in normalised error units."""
        moved_x = moved_y = 0.0
        history = self._motion_history
        for index, (sent_at, velocity_x, velocity_y) in enumerate(history):
            until = history[index + 1][0] if index + 1 < len(history) else end
            overlap = min(until, end) - max(sent_at, start)
            if overlap > 0:
                moved_x += velocity_x * overlap
                moved_y += velocity_y * overlap
        return moved_x, moved_y

    def _delay_corrected_error(self, error_x: float, error_y: float, now: float) -> tuple[float, float]:
        """Estimate the error as it is now, not as the delayed picture shows it.

        The picture is `delay` seconds old. Since then the target kept moving
        (prediction) and the camera carried out commands that are not visible
        yet (stop look-ahead).
        """
        delay = max(self._predict_target_seconds, self._stop_lookahead_seconds)
        while len(self._motion_history) > 1 and self._motion_history[1][0] < now - delay - 1.0:
            self._motion_history.pop(0)
        if self._predict_target_seconds:
            previous = self._previous_error
            self._previous_error = (now, error_x, error_y)
            if previous is not None and 0.0 < now - previous[0] < 0.5:
                elapsed = now - previous[0]
                # Camera motion visible in this picture interval was commanded one delay earlier.
                seen_x, seen_y = self._camera_motion(previous[0] - delay, now - delay)
                raw_x = (error_x - previous[1] + seen_x) / elapsed
                raw_y = (error_y - previous[2] + seen_y) / elapsed
                if abs(error_x - previous[1]) > 0.4 or abs(error_y - previous[2]) > 0.4:
                    self._target_velocity = (0.0, 0.0)  # a jump is a new box, not motion
                else:
                    blend = min(1.0, elapsed / 0.3)
                    self._target_velocity = (
                        max(-1.5, min(1.5, self._target_velocity[0] + blend * (raw_x - self._target_velocity[0]))),
                        max(-1.5, min(1.5, self._target_velocity[1] + blend * (raw_y - self._target_velocity[1]))),
                    )
            else:
                self._target_velocity = (0.0, 0.0)
            error_x += self._target_velocity[0] * self._predict_target_seconds
            if self._predict_axes == "both":
                # The vertical estimate is mostly box-height jitter; "pan" leaves it out.
                error_y += self._target_velocity[1] * self._predict_target_seconds
        if self._stop_lookahead_seconds:
            pending_x, pending_y = self._camera_motion(now - self._stop_lookahead_seconds, now)
            error_x -= pending_x
            error_y -= pending_y
        return error_x, error_y

    # [room_01 수정] 신규 메서드. uniform_error_scale이면 세로 오차도 화면 반폭으로 나눔.
    # 기존: 세로는 반높이(180px)로 나눠 같은 픽셀 오차가 가로보다 1.8배 크게 계산됨
    # -> tilt deadzone이 18px에 불과해 박스 흔들림에도 위아래로 움직였음.
    def _normalized_error(
        self,
        target: BoundingBox,
        width: float,
        height: float,
        now: float | None = None,
    ) -> tuple[float, float]:
        target_x, target_y = target.center
        # A wide frame is much shorter than it is wide. Scaling the tilt error
        # by the half height makes the same pixel offset look almost twice as
        # large vertically, so tilt reacts to jitter that pan ignores and
        # overshoots its narrow deadzone. A uniform scale measures both axes
        # in the same pixels.
        vertical_scale = width if self._uniform_error_scale else height
        if now is None or not (self._predict_target_seconds or self._stop_lookahead_seconds):
            return (
                (target_x - (width / 2)) / max(width / 2, 1.0),
                (target_y - (height / 2)) / max(vertical_scale / 2, 1.0),
            )
        # Optional prediction / stop look-ahead (--qon-predict-target-seconds,
        # --qon-stop-lookahead-seconds) works in half-frame units on both axes;
        # the tilt error is rescaled afterwards.
        error_x, error_y = self._delay_corrected_error(
            (target_x - (width / 2)) / max(width / 2, 1.0),
            (target_y - (height / 2)) / max(height / 2, 1.0),
            now,
        )
        return error_x, error_y * max(height / 2, 1.0) / max(vertical_scale / 2, 1.0)

    # [room_01 수정] 신규 메서드. 현재 명령이 지연 시간 동안 더 줄일 오차
    # (속도 x 속도당 이동률 x 지연)를 측정 오차에서 뺌. 방향 반전은 만들지 않음.
    def _compensated_error(
        self,
        error: float,
        positive_command: str,
        negative_command: str,
        rate_per_speed: float,
    ) -> float:
        """Error expected once the motion still in the pipeline has shown up."""
        if self._response_delay_seconds <= 0 or self._moving_since is None:
            return error
        if self._last_command == positive_command:
            direction = 1.0
        elif self._last_command == negative_command:
            direction = -1.0
        else:
            return error
        if error * direction <= 0:
            return error
        pending_seconds = min(
            self._response_delay_seconds,
            max(0.0, self._now() - self._moving_since),
        )
        pending_travel = rate_per_speed * self._last_speed * pending_seconds
        # A target that keeps walking does not get closer just because the
        # camera moves, so only part of a large error may be written off.
        pending_travel = min(
            pending_travel,
            self._MAX_COMPENSATED_FRACTION * abs(error),
        )
        return error - direction * pending_travel

    def _stop(self) -> None:
        # [room_01 수정] 정지 시 지연 보상 상태 초기화.
        self._moving_since = None
        if self._last_command != "ptzstop":
            self._send("ptzstop", 10, 10)
        self._last_command = "ptzstop"
        self._last_speed = 0
        self._last_sent_at = self._now()

    def _send(self, command: str, speed_x: int, speed_y: int) -> None:
        if self._level_speeds:
            self._motion_history.append((self._now(), *self._level_velocity(command, speed_x, speed_y)))
        if not self._async_commands:
            self._send_with_warning(command, speed_x, speed_y)
            return
        with self._condition:
            self._pending_command = (command, speed_x, speed_y)
            self._condition.notify()

    def _send_with_warning(self, command: str, speed_x: int, speed_y: int | str | None) -> bool:
        try:
            self._send_sync(command, speed_x, speed_y)
        except Exception as exc:  # pragma: no cover - depends on camera/network failures.
            self.last_error = exc
            logging.warning("Failed to send Qon PTZ command %s to %s: %s", command, self._camera_url, exc)
            return False
        return True

    def _send_sync(self, command: str, speed_x: int, speed_y: int | str | None) -> None:
        url = f"{self._camera_url}/cgi-bin/ptzctrl.cgi?ptzcmd&{command}&{speed_x}"
        if speed_y is not None:  # continuous zoom commands take a single speed
            url += f"&{speed_y}"
        if self._log_commands:
            logging.info("Qon HTTP PTZ send GET %s", url)
        request = Request(
            url,
            headers={"Accept": "*/*", "Content-Type": "application/x-www-form-urlencoded;charset=utf-8"},
            method="GET",
        )
        self._add_auth(request)
        open_fn = self._opener.open if hasattr(self._opener, "open") else self._opener
        with open_fn(request, timeout=self._timeout_seconds) as response:
            response.read()
            if self._log_commands:
                logging.info(
                    "Qon HTTP PTZ response command=%s status=%s",
                    command,
                    getattr(response, "status", "unknown"),
                )

    def _send_loop(self) -> None:
        while True:
            with self._condition:
                while self._pending_command is None and self._pending_zoom is None and not self._closed:
                    deadline = self._zoom_deadline()
                    if deadline is None:
                        self._condition.wait()
                        continue
                    remaining = deadline - self._now()
                    if remaining <= 0:
                        # Burst limit reached without a new request (e.g. the main loop stalled).
                        self._last_zoom = ("stop", 0)
                        self._zoom_burst_started_at = None
                        self._zoom_rest_until = self._now() + self._zoom_burst_rest_seconds
                        self._pending_zoom = ("zoomstop", 0, None)
                        break
                    self._condition.wait(timeout=remaining)
                if self._pending_command is None and self._pending_zoom is None and self._closed:
                    return
                pending, self._pending_command = self._pending_command, None
                pending_zoom, self._pending_zoom = self._pending_zoom, None
            if pending is not None:
                self._send_with_warning(*pending)
            if pending_zoom is not None:
                self._send_zoom(*pending_zoom)

    def _apply_supervisor_actuator_mode(self) -> None:
        try:
            self._post_param(
                "write_path",
                {
                    "cururl": "http://",
                    "path": self._TRACK_CONFIG_PATH,
                    "common.track": "0",
                    "tracking.auto_zoom": "0",
                    "tracking.auto_tilt": "0",
                    "common.debug_mode": "3",
                },
            )
            self._post_param(
                "post_visca",
                {
                    "cururl": "http://",
                    "len": str(len(self._TRACK_REFRESH_VISCA)),
                    "visca": ",".join(f"0x{value:02X}" for value in self._TRACK_REFRESH_VISCA),
                },
            )
            response_body = self._post_param(
                "get_path",
                {"path": self._TRACK_CONFIG_PATH},
                content_type="text/plain;charset=UTF-8",
            )
            self._verify_supervisor_actuator_body(response_body)
            logging.info("Qon supervisor actuator mode applied: %s", response_body.strip())
        except Exception as exc:
            self.last_error = exc
            logging.warning("Failed to apply Qon supervisor actuator mode to %s: %s", self._camera_url, exc)

    def _post_param(
        self,
        action: str,
        values: dict[str, str],
        content_type: str = "application/x-www-form-urlencoded;charset=UTF-8",
    ) -> str:
        url = f"{self._camera_url}/cgi-bin/param.cgi?{action}"
        data = urlencode(values).encode("utf-8")
        if self._log_commands:
            logging.info("Qon HTTP param send POST %s data=%s", url, data.decode("utf-8"))
        request = Request(
            url,
            data=data,
            headers={"Accept": "*/*", "Content-Type": content_type},
            method="POST",
        )
        self._add_auth(request)
        open_fn = self._opener.open if hasattr(self._opener, "open") else self._opener
        with open_fn(request, timeout=self._timeout_seconds) as response:
            body = response.read().decode("utf-8", errors="replace")
            if self._log_commands:
                logging.info(
                    "Qon HTTP param response action=%s status=%s body=%s",
                    action,
                    getattr(response, "status", "unknown"),
                    body.strip(),
                )
            return body

    def _add_auth(self, request: Request) -> None:
        if self._username is None or self._password is None or self._auth_mode != "basic":
            return
        token = base64.b64encode(f"{self._username}:{self._password}".encode("utf-8")).decode("ascii")
        request.add_header("Authorization", f"Basic {token}")

    def _build_opener(self):
        if self._username is None or self._password is None or self._auth_mode == "none":
            return urlopen
        password_mgr = HTTPPasswordMgrWithDefaultRealm()
        password_mgr.add_password(None, self._camera_url, self._username, self._password)
        if self._auth_mode == "digest":
            return build_opener(HTTPDigestAuthHandler(password_mgr))
        if self._auth_mode == "basic":
            return build_opener(HTTPBasicAuthHandler(password_mgr))
        raise ValueError("Qon auth_mode must be one of none, basic, digest")

    def _verify_supervisor_actuator_body(self, body: str) -> None:
        expected = {
            "common.track": "0",
            "tracking.auto_zoom": "0",
            "tracking.auto_tilt": "0",
            "common.debug_mode": "3",
        }
        missing = []
        for key, value in expected.items():
            pattern = rf"{re.escape(key)}\s*=\s*\"?{re.escape(value)}\"?"
            if re.search(pattern, body) is None:
                missing.append(f"{key}={value}")
        if missing:
            logging.warning("Qon supervisor actuator read-back missing expected values: %s", ", ".join(missing))

    def _now(self) -> float:
        if self._now_fn is not None:
            return self._now_fn()
        import time

        return time.monotonic()
