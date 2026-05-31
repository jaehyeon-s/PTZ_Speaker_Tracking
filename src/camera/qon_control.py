"""Confirmed HTTP control surface for Qon4K6012XN tracking mode."""

from __future__ import annotations

import argparse
import base64
import os
import re
from dataclasses import dataclass
from enum import IntEnum
from urllib.parse import urlencode
from urllib.request import Request, urlopen


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
        timeout: float = 5.0,
        opener=urlopen,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.username = username
        self.password = password
        self.timeout = timeout
        self._opener = opener

    def read_track_config(self) -> QonResponse:
        return self._post("get_path", {"path": TRACK_CONFIG_PATH}, content_type="text/plain;charset=UTF-8")

    def get_mode(self) -> TrackingMode | None:
        response = self.read_track_config()
        return parse_mode(response.body)

    def set_mode(self, mode: TrackingMode) -> QonResponse:
        return self._post(
            "write_path",
            {"cururl": "http://", "path": TRACK_CONFIG_PATH, "common.track": str(int(mode))},
        )

    def enable_auto_tracking(self) -> QonResponse:
        return self.set_mode(TrackingMode.TRACKING)

    def enable_zone_tracking(self) -> QonResponse:
        return self.set_mode(TrackingMode.ZONE)

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
            token = base64.b64encode(f"{self.username}:{self.password}".encode("utf-8")).decode("ascii")
            request.add_header("Authorization", f"Basic {token}")
        with self._opener(request, timeout=self.timeout) as response:
            return QonResponse(int(getattr(response, "status", 200)), response.read().decode("utf-8", errors="replace"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Read or set Qon4K6012XN tracking mode")
    parser.add_argument("--camera-url", required=True, help="Camera origin, for example http://camera-address")
    parser.add_argument("--username", default=None)
    parser.add_argument("--password-env", default=None, help="Environment variable holding camera password")
    parser.add_argument("command", choices=("status", "tracking", "zone"))
    args = parser.parse_args()
    password = os.environ.get(args.password_env) if args.password_env else None
    client = QonTrackingControl(args.camera_url, args.username, password)
    if args.command == "status":
        response = client.read_track_config()
        mode = parse_mode(response.body)
        print(mode.name if mode is not None else response.body)
        return 0
    response = client.set_mode(TrackingMode.TRACKING if args.command == "tracking" else TrackingMode.ZONE)
    print(f"HTTP {response.status}: requested {args.command} mode")
    return 0


def parse_mode(body: str) -> TrackingMode | None:
    match = re.search(r"common\.track\s*=\s*([01])", body)
    return TrackingMode(int(match.group(1))) if match else None


if __name__ == "__main__":
    raise SystemExit(main())
