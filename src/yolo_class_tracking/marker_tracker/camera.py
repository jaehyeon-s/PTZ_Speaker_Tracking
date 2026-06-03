from __future__ import annotations

import logging
from typing import Protocol

from .core import BoundingBox


class CameraTrackingAdapter(Protocol):
    def handoff(self, target: BoundingBox) -> None: ...

    def release(self) -> None: ...


class LoggingCameraAdapter:
    """Temporary adapter until the Qon camera control API is available."""

    def handoff(self, target: BoundingBox) -> None:
        logging.info("PTZ handoff requested: person_box=%s", target)

    def release(self) -> None:
        logging.info("PTZ tracking release requested: restore camera default")

