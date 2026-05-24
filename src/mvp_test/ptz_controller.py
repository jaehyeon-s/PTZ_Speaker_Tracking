"""PTZ simulator and command calculation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from target_state import BBox


@dataclass(frozen=True)
class PTZCommand:
    pan: str = "STOP"
    tilt: str = "STOP"
    error_x: float = 0.0
    error_y: float = 0.0

    def as_text(self) -> str:
        if self.pan == "STOP" and self.tilt == "STOP":
            return "STOP"
        parts = []
        if self.pan != "STOP":
            parts.append(self.pan)
        if self.tilt != "STOP":
            parts.append(self.tilt)
        return "+".join(parts)


class PTZSimulator:
    """Computes coarse pan/tilt commands from bbox center and prints changes."""

    def __init__(self, dead_zone_ratio: float = 0.12, print_changes_only: bool = True) -> None:
        self.dead_zone_ratio = dead_zone_ratio
        self.print_changes_only = print_changes_only
        self._last_text: Optional[str] = None

    def command_for_bbox(self, bbox: BBox, frame_shape: tuple[int, int, int]) -> PTZCommand:
        frame_h, frame_w = frame_shape[:2]
        x, y, w, h = bbox
        target_x = x + w / 2.0
        target_y = y + h / 2.0
        error_x = (target_x - frame_w / 2.0) / max(frame_w / 2.0, 1.0)
        error_y = (target_y - frame_h / 2.0) / max(frame_h / 2.0, 1.0)

        dead_x = self.dead_zone_ratio
        dead_y = self.dead_zone_ratio
        pan = "STOP"
        tilt = "STOP"
        if error_x < -dead_x:
            pan = "LEFT"
        elif error_x > dead_x:
            pan = "RIGHT"
        if error_y < -dead_y:
            tilt = "UP"
        elif error_y > dead_y:
            tilt = "DOWN"

        return PTZCommand(pan=pan, tilt=tilt, error_x=error_x, error_y=error_y)

    def send(self, command: PTZCommand) -> None:
        text = command.as_text()
        if not self.print_changes_only or text != self._last_text:
            print(f"PTZ_SIM {text} err=({command.error_x:.2f},{command.error_y:.2f})")
            self._last_text = text


class HardwarePTZController:
    """Placeholder for real PTZ camera integration."""

    def send(self, command: PTZCommand) -> None:
        # TODO: Replace this with VISCA/ONVIF/vendor SDK commands for real hardware.
        print(f"PTZ_HW_TODO {command.as_text()} err=({command.error_x:.2f},{command.error_y:.2f})")
