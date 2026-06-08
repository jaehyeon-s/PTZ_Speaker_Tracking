"""Confirmed HTTP control surface for Qon4K6012XN tracking mode."""

from __future__ import annotations

import argparse
import base64
import os
import re
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
    `common.track=1` for tracking and `common.track=0` for zone mode.
    The additional captured VISCA posts occurred in both mode transitions and
    are intentionally not sent here until their purpose is identified.
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
        choices=("status", "tracking", "zone", "stop", "home", "zoomout", "zoomin", "debug-bbox", "debug-off", "hint-on", "hint-off"),
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
    else:
        response = client.set_tracking_hint(False)
    print(f"HTTP {response.status}: requested {args.command} mode")
    return 0


def parse_mode(body: str) -> TrackingMode | None:
    match = re.search(r"common\.track\s*=\s*([01])", body)
    return TrackingMode(int(match.group(1))) if match else None


if __name__ == "__main__":
    raise SystemExit(main())
