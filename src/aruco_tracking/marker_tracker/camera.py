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
from typing import Protocol

from .core import BoundingBox


class CameraTrackingAdapter(Protocol):
    def update(self, target: BoundingBox | None, frame_shape: tuple[int, ...]) -> None: ...

    def handoff(self, target: BoundingBox) -> None: ...

    def release(self) -> None: ...

    def status(self) -> str: ...

    def close(self) -> None: ...


class LoggingCameraAdapter:
    """Temporary adapter until the Qon camera control API is available."""

    def update(self, target: BoundingBox | None, frame_shape: tuple[int, ...]) -> None:
        del target, frame_shape

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

    def update(self, target: BoundingBox | None, frame_shape: tuple[int, ...]) -> None:
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

    def __init__(
        self,
        camera_url: str,
        username: str | None = None,
        password: str | None = None,
        min_speed: int = 2,
        max_speed: int = 10,
        deadzone: float = 0.12,
        command_ttl_seconds: float = 0.35,
        timeout_seconds: float = 1.0,
        configure_supervisor_actuator: bool = True,
        auth_mode: str = "basic",
        async_commands: bool = True,
        log_commands: bool = False,
        close_join_timeout_seconds: float = 1.0,
        now_fn=None,
        opener=None,
    ) -> None:
        if not camera_url:
            raise ValueError("Qon camera URL is required")
        if min_speed < 1 or max_speed < 1:
            raise ValueError("Qon PTZ speeds must be positive")
        if deadzone < 0 or command_ttl_seconds < 0:
            raise ValueError("Qon PTZ deadzone and TTL must be non-negative")
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
        self._command_ttl_seconds = command_ttl_seconds
        self._timeout_seconds = timeout_seconds
        self._configure_supervisor_actuator = configure_supervisor_actuator
        self._async_commands = async_commands
        self._log_commands = log_commands
        self._close_join_timeout_seconds = close_join_timeout_seconds
        self._now_fn = now_fn
        self._opener = opener or self._build_opener()
        self._last_command = "ptzstop"
        self._last_speed = 0
        self._last_sent_at = 0.0
        self.last_error: Exception | None = None
        self._pending_command: tuple[str, int, int] | None = None
        self._closed = False
        self._condition = threading.Condition()
        self._worker: threading.Thread | None = None
        if self._async_commands:
            self._worker = threading.Thread(target=self._send_loop, daemon=True)
            self._worker.start()
        logging.info(
            "Qon HTTP camera control enabled: url=%s min_speed=%s max_speed=%s deadzone=%s async=%s auth=%s log_commands=%s",
            self._camera_url,
            self._min_speed,
            self._max_speed,
            self._deadzone,
            self._async_commands,
            self._auth_mode,
            self._log_commands,
        )
        if self._configure_supervisor_actuator:
            self._apply_supervisor_actuator_mode()

    def update(self, target: BoundingBox | None, frame_shape: tuple[int, ...]) -> None:
        command, speed = self._command_for_target(target, frame_shape)
        if command == "ptzstop":
            self._stop()
            return
        now = self._now()
        if (
            command != self._last_command
            or speed != self._last_speed
            or now - self._last_sent_at >= self._command_ttl_seconds
        ):
            self._send(command, speed, speed)
            self._last_command = command
            self._last_speed = speed
            self._last_sent_at = now

    def handoff(self, target: BoundingBox) -> None:
        logging.info("PTZ Qon HTTP tracking target selected: person_box=%s", target)

    def release(self) -> None:
        logging.info("PTZ Qon HTTP tracking release requested: stopping pan/tilt")
        self._stop()

    def status(self) -> str:
        if self._last_command == "ptzstop":
            return "ptzstop"
        return f"{self._last_command}:{self._last_speed}"

    def close(self) -> None:
        self._stop()
        if self._worker is not None:
            with self._condition:
                self._closed = True
                self._condition.notify()
            self._worker.join(timeout=self._close_join_timeout_seconds)
        self._send_with_warning("ptzstop", 10, 10)

    def _command_for_target(
        self, target: BoundingBox | None, frame_shape: tuple[int, ...]
    ) -> tuple[str, int]:
        if target is None or len(frame_shape) < 2:
            return "ptzstop", 0
        height, width = frame_shape[:2]
        if width <= 0 or height <= 0:
            return "ptzstop", 0
        target_x, target_y = target.center
        error_x = (target_x - (width / 2)) / max(width / 2, 1.0)
        error_y = (target_y - (height / 2)) / max(height / 2, 1.0)
        if abs(error_x) <= self._deadzone and abs(error_y) <= self._deadzone:
            return "ptzstop", 0
        if abs(error_x) >= abs(error_y):
            direction = "right" if error_x > 0 else "left"
            magnitude = abs(error_x)
        else:
            direction = "down" if error_y > 0 else "up"
            magnitude = abs(error_y)
        speed_span = self._max_speed - self._min_speed
        scaled = min(
            1.0,
            max(0.0, (magnitude - self._deadzone) / max(1.0 - self._deadzone, 1e-6)),
        )
        return direction, int(round(self._min_speed + scaled * speed_span))

    def _stop(self) -> None:
        if self._last_command != "ptzstop":
            self._send("ptzstop", 10, 10)
        self._last_command = "ptzstop"
        self._last_speed = 0
        self._last_sent_at = self._now()

    def _send(self, command: str, speed_x: int, speed_y: int) -> None:
        if not self._async_commands:
            self._send_with_warning(command, speed_x, speed_y)
            return
        with self._condition:
            self._pending_command = (command, speed_x, speed_y)
            self._condition.notify()

    def _send_with_warning(self, command: str, speed_x: int, speed_y: int) -> None:
        try:
            self._send_sync(command, speed_x, speed_y)
        except Exception as exc:  # pragma: no cover - depends on camera/network failures.
            self.last_error = exc
            logging.warning("Failed to send Qon PTZ command %s to %s: %s", command, self._camera_url, exc)

    def _send_sync(self, command: str, speed_x: int, speed_y: int) -> None:
        url = f"{self._camera_url}/cgi-bin/ptzctrl.cgi?ptzcmd&{command}&{speed_x}&{speed_y}"
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
                while self._pending_command is None and not self._closed:
                    self._condition.wait()
                if self._pending_command is None and self._closed:
                    return
                command, speed_x, speed_y = self._pending_command
                self._pending_command = None
            self._send_with_warning(command, speed_x, speed_y)

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
