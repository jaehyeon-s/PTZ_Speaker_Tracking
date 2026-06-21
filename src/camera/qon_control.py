"""Confirmed HTTP control surface for Qon4K6012XN tracking mode."""

from __future__ import annotations

import argparse
import base64
import os
import re
import threading
import time
from dataclasses import dataclass
from enum import IntEnum
from urllib.parse import urlencode
from urllib.request import (
    HTTPBasicAuthHandler,
    HTTPDigestAuthHandler,
    HTTPPasswordMgrWithDefaultRealm,
    Request,
    build_opener,
    urlopen,
)


TRACK_CONFIG_PATH = "/data/track.conf"
TRACK_REFRESH_VISCA = [0x81, 0x0A, 0x01, 0x04, 0x1D, 0x17, 0xFF]


class TrackingMode(IntEnum):
    ZONE = 0
    TRACKING = 1


@dataclass(frozen=True)
class QonResponse:
    status: int
    body: str


class QonTrackingControl:
    """Read and set the tracking/zone mode observed in the camera web UI.

    Network capture established that `/data/track.conf` uses
    `common.track=1` for tracking and `common.track=0` for tracking off. Field
    testing showed that settings requiring runtime application should be
    followed by the captured VISCA refresh and verified by read-back.
    """

    def __init__(
        self,
        base_url: str,
        username: str | None = None,
        password: str | None = None,
        auth_mode: str = "basic",
        timeout: float = 5.0,
        opener=None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.username = username
        self.password = password
        self.auth_mode = auth_mode
        self.timeout = timeout
        self._opener = opener or self._build_opener()

    def read_track_config(self) -> QonResponse:
        return self._post("get_path", {"path": TRACK_CONFIG_PATH}, content_type="text/plain;charset=UTF-8")

    def get_mode(self) -> TrackingMode | None:
        response = self.read_track_config()
        return parse_mode(response.body)

    def set_mode(self, mode: TrackingMode) -> QonResponse:
        return self.write_track_config("common.track", str(int(mode)))

    def enable_auto_tracking(self) -> QonResponse:
        return self.set_mode(TrackingMode.TRACKING)

    def enable_zone_tracking(self) -> QonResponse:
        return self.set_mode(TrackingMode.ZONE)

    def write_track_config(self, key: str, value: str | int) -> QonResponse:
        return self._post("write_path", {"cururl": "http://", "path": TRACK_CONFIG_PATH, key: str(value)})

    def write_track_config_values(self, values: dict[str, str | int]) -> QonResponse:
        payload = {"cururl": "http://", "path": TRACK_CONFIG_PATH}
        payload.update({key: str(value) for key, value in values.items()})
        return self._post("write_path", payload)

    def refresh_tracking_runtime(self) -> QonResponse:
        return self.post_visca(TRACK_REFRESH_VISCA)

    def apply_track_config(self, values: dict[str, str | int], refresh: bool = True) -> QonResponse:
        self.write_track_config_values(values)
        if refresh:
            self.refresh_tracking_runtime()
        return self.read_track_config()

    def set_supervisor_actuator_mode(self) -> QonResponse:
        """Disable camera-owned tracking/PTZ motion for Pi supervisor control."""
        return self.apply_track_config(
            {
                "common.track": 0,
                "tracking.auto_zoom": 0,
                "tracking.auto_tilt": 0,
                "common.debug_mode": 3,
            }
        )

    def set_humanoid_frame(self, mode: int) -> QonResponse:
        """Set observed humanoid frame mode: 2=debug bbox, 3=off, 4=default."""
        if mode not in (2, 3, 4):
            raise ValueError("humanoid frame mode must be one of 2, 3, 4")
        return self.write_track_config("common.debug_mode", mode)

    def set_tracking_hint(self, enabled: bool) -> QonResponse:
        return self.write_track_config("common.osd_mode", 1 if enabled else 0)

    def set_presenter_mode(self) -> QonResponse:
        return self.write_track_config("common.track_mode", "tracking")

    def ptz_command(self, command: str, speed_x: int = 10, speed_y: int | None = 10) -> QonResponse:
        if speed_y is None:
            action = f"ptzcmd&{command}&{speed_x}"
        else:
            action = f"ptzcmd&{command}&{speed_x}&{speed_y}"
        return self._get_ptz(action)

    def stop(self) -> QonResponse:
        return self.ptz_command("ptzstop", 10, 10)

    def home(self) -> QonResponse:
        return self.ptz_command("home", 10, 10)

    def zoom_out(self, speed: int = 5) -> QonResponse:
        return self.ptz_command("zoomout", speed, None)

    def zoom_in(self, speed: int = 5) -> QonResponse:
        return self.ptz_command("zoomin", speed, None)

    def zoom_stop(self, speed: int = 5) -> QonResponse:
        return self.ptz_command("zoomstop", speed, None)

    def post_visca(self, values: list[int]) -> QonResponse:
        visca = ",".join(f"0x{value:02X}" for value in values)
        return self._post("post_visca", {"cururl": "http://", "len": str(len(values)), "visca": visca})

    def _post(
        self, action: str, values: dict[str, str], content_type: str = "application/x-www-form-urlencoded;charset=UTF-8"
    ) -> QonResponse:
        request = Request(
            f"{self.base_url}/cgi-bin/param.cgi?{action}",
            data=urlencode(values).encode("utf-8"),
            headers={"Accept": "*/*", "Content-Type": content_type},
            method="POST",
        )
        if self.username is not None and self.password is not None:
            if self.auth_mode == "digest":
                pass
            elif self.auth_mode == "none":
                pass
            else:
                token = base64.b64encode(f"{self.username}:{self.password}".encode("utf-8")).decode("ascii")
                request.add_header("Authorization", f"Basic {token}")
        return self._open(request)

    def _get_ptz(self, action: str) -> QonResponse:
        request = Request(
            f"{self.base_url}/cgi-bin/ptzctrl.cgi?{action}",
            headers={"Accept": "*/*", "Content-Type": "application/x-www-form-urlencoded;charset=utf-8"},
            method="GET",
        )
        if self.username is not None and self.password is not None and self.auth_mode == "basic":
            token = base64.b64encode(f"{self.username}:{self.password}".encode("utf-8")).decode("ascii")
            request.add_header("Authorization", f"Basic {token}")
        return self._open(request)

    def _open(self, request: Request) -> QonResponse:
        opener = self._opener
        open_fn = opener.open if hasattr(opener, "open") else opener
        with open_fn(request, timeout=self.timeout) as response:
            return QonResponse(int(getattr(response, "status", 200)), response.read().decode("utf-8", errors="replace"))

    def _build_opener(self):
        if self.username is None or self.password is None or self.auth_mode == "none":
            return urlopen
        password_mgr = HTTPPasswordMgrWithDefaultRealm()
        password_mgr.add_password(None, self.base_url, self.username, self.password)
        if self.auth_mode == "digest":
            return build_opener(HTTPDigestAuthHandler(password_mgr))
        if self.auth_mode == "basic":
            return build_opener(HTTPBasicAuthHandler(password_mgr))
        raise ValueError("--auth-mode must be one of none, basic, digest")


def main() -> int:
    parser = argparse.ArgumentParser(description="Read or set Qon4K6012XN tracking mode")
    parser.add_argument("--camera-url", required=True, help="Camera origin, for example http://camera-address")
    parser.add_argument("--username", default=None)
    parser.add_argument("--auth-mode", choices=("none", "basic", "digest"), default="basic")
    parser.add_argument("--password-env", default=None, help="Environment variable holding camera password")
    parser.add_argument(
        "command",
        choices=(
            "status",
            "tracking",
            "zone",
            "stop",
            "home",
            "zoomout",
            "zoomin",
            "debug-bbox",
            "debug-off",
            "hint-on",
            "hint-off",
            "supervisor-actuator",
        ),
    )
    args = parser.parse_args()
    password = os.environ.get(args.password_env) if args.password_env else None
    client = QonTrackingControl(args.camera_url, args.username, password, auth_mode=args.auth_mode)
    if args.command == "status":
        response = client.read_track_config()
        mode = parse_mode(response.body)
        print(mode.name if mode is not None else response.body)
        return 0
    if args.command == "tracking":
        response = client.enable_auto_tracking()
    elif args.command == "zone":
        response = client.enable_zone_tracking()
    elif args.command == "stop":
        response = client.stop()
    elif args.command == "home":
        response = client.home()
    elif args.command == "zoomout":
        response = client.zoom_out()
        client.zoom_stop()
    elif args.command == "zoomin":
        response = client.zoom_in()
        client.zoom_stop()
    elif args.command == "debug-bbox":
        response = client.set_humanoid_frame(2)
    elif args.command == "debug-off":
        response = client.set_humanoid_frame(3)
    elif args.command == "hint-on":
        response = client.set_tracking_hint(True)
    elif args.command == "supervisor-actuator":
        response = client.set_supervisor_actuator_mode()
    else:
        response = client.set_tracking_hint(False)
    print(f"HTTP {response.status}: requested {args.command} mode")
    return 0


def parse_mode(body: str) -> TrackingMode | None:
    match = re.search(r"common\.track\s*=\s*([01])", body)
    return TrackingMode(int(match.group(1))) if match else None


class QonVelocityPTZController:
    """Closed-loop PTZ driver for Qon's direction/speed command model."""

    def __init__(
        self,
        control: QonTrackingControl,
        dead_zone_ratio: float = 0.12,
        min_speed: int = 2,
        max_speed: int = 10,
        command_ttl_seconds: float = 0.35,
        async_commands: bool = False,
        close_join_timeout_seconds: float = 1.0,
        zoom_enabled: bool = False,
        target_height_ratio: float = 0.6,
        zoom_tolerance: float = 0.08,
        zoom_speed: int = 3,
    ) -> None:
        self.control = control
        self.dead_zone_ratio = dead_zone_ratio
        self.min_speed = min_speed
        self.max_speed = max(min_speed, max_speed)
        self.command_ttl_seconds = command_ttl_seconds
        self.zoom_enabled = zoom_enabled
        self.target_height_ratio = target_height_ratio
        self.zoom_tolerance = zoom_tolerance
        self.zoom_speed = max(1, zoom_speed)
        self.async_commands = async_commands
        self.last_command = "ptzstop"
        self.last_speed = 0
        self.last_sent_at = 0.0
        self.last_error: Exception | None = None
        self.close_join_timeout_seconds = close_join_timeout_seconds
        self._pending_command: tuple[str, int, int | None] | None = None
        self._closed = False
        self._condition = threading.Condition()
        self._worker: threading.Thread | None = None
        if self.async_commands:
            self._worker = threading.Thread(target=self._send_loop, daemon=True)
            self._worker.start()

    def follow_bbox(self, bbox: tuple[int, int, int, int] | None, frame_shape) -> str:
        if bbox is None:
            self.stop()
            return "ptzstop"
        command, speed = self._command_for_bbox(bbox, frame_shape)
        speed_y: int | None = speed
        if command == "ptzstop":
            # Centered on pan/tilt; adjust zoom so the subject fills the target height.
            command, speed = self._zoom_for_bbox(bbox, frame_shape)
            speed_y = None
        if command == "ptzstop":
            self.stop()
            return command
        now = time.monotonic()
        if command != self.last_command or speed != self.last_speed or now - self.last_sent_at >= self.command_ttl_seconds:
            self._send(command, speed, speed_y)
            self.last_command = command
            self.last_speed = speed
            self.last_sent_at = now
        return f"{command}:{speed}"

    def _zoom_for_bbox(self, bbox: tuple[int, int, int, int], frame_shape) -> tuple[str, int]:
        if not self.zoom_enabled:
            return "ptzstop", 0
        frame_h = frame_shape[0]
        _, y1, _, y2 = bbox
        height_ratio = (y2 - y1) / max(float(frame_h), 1.0)
        if height_ratio < self.target_height_ratio - self.zoom_tolerance:
            return "zoomin", self.zoom_speed
        if height_ratio > self.target_height_ratio + self.zoom_tolerance:
            return "zoomout", self.zoom_speed
        return "ptzstop", 0

    def stop(self) -> None:
        if self.last_command != "ptzstop":
            self._send("ptzstop", 10, 10)
        self.last_command = "ptzstop"
        self.last_speed = 0
        self.last_sent_at = time.monotonic()

    def close(self) -> None:
        self.stop()
        if self._worker is not None:
            with self._condition:
                self._closed = True
                self._condition.notify()
            self._worker.join(timeout=self.close_join_timeout_seconds)
        try:
            self.control.stop()
        except Exception as exc:
            self.last_error = exc

    def _command_for_bbox(self, bbox: tuple[int, int, int, int], frame_shape) -> tuple[str, int]:
        frame_h, frame_w = frame_shape[:2]
        x1, y1, x2, y2 = bbox
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        error_x = (cx - frame_w / 2.0) / max(frame_w / 2.0, 1.0)
        error_y = (cy - frame_h / 2.0) / max(frame_h / 2.0, 1.0)
        if abs(error_x) <= self.dead_zone_ratio and abs(error_y) <= self.dead_zone_ratio:
            return "ptzstop", 0
        if abs(error_x) >= abs(error_y):
            direction = "right" if error_x > 0 else "left"
            magnitude = abs(error_x)
        else:
            direction = "down" if error_y > 0 else "up"
            magnitude = abs(error_y)
        speed_span = self.max_speed - self.min_speed
        scaled = min(1.0, max(0.0, (magnitude - self.dead_zone_ratio) / max(1.0 - self.dead_zone_ratio, 1e-6)))
        return direction, int(round(self.min_speed + scaled * speed_span))

    def _send(self, command: str, speed_x: int, speed_y: int | None) -> None:
        if not self.async_commands:
            self.control.ptz_command(command, speed_x, speed_y)
            return
        with self._condition:
            self._pending_command = (command, speed_x, speed_y)
            self._condition.notify()

    def _send_loop(self) -> None:
        while True:
            with self._condition:
                while self._pending_command is None and not self._closed:
                    self._condition.wait()
                if self._pending_command is None and self._closed:
                    return
                command, speed_x, speed_y = self._pending_command
                self._pending_command = None
            try:
                self.control.ptz_command(command, speed_x, speed_y)
            except Exception as exc:  # pragma: no cover - depends on camera/network failures.
                self.last_error = exc


if __name__ == "__main__":
    raise SystemExit(main())
