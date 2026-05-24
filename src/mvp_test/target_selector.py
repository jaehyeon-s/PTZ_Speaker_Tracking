"""Select and maintain the active target for the PTZ tracking MVP."""

from __future__ import annotations

import math
from typing import Iterable, List, Optional

from target_state import ActiveTarget, BBox, DemoConfig, MarkerDetection, PersonDetection


def point_in_bbox(point: tuple[float, float], bbox: BBox) -> bool:
    px, py = point
    x, y, w, h = bbox
    return x <= px <= x + w and y <= py <= y + h


def bbox_iou(a: BBox, b: BBox) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1 = max(ax, bx)
    y1 = max(ay, by)
    x2 = min(ax + aw, bx + bw)
    y2 = min(ay + ah, by + bh)
    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter = inter_w * inter_h
    union = aw * ah + bw * bh - inter
    return inter / union if union else 0.0


def center_distance(a: BBox, b: BBox) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ac = (ax + aw / 2.0, ay + ah / 2.0)
    bc = (bx + bw / 2.0, by + bh / 2.0)
    return math.hypot(ac[0] - bc[0], ac[1] - bc[1])


class TargetSelector:
    """Owns target acquisition and simple bbox-based reacquisition."""

    def __init__(self, config: DemoConfig) -> None:
        self.config = config
        self._next_target_id = 1

    def reset_ids(self) -> None:
        self._next_target_id = 1

    def acquire_from_marker(
        self,
        people: Iterable[PersonDetection],
        markers: Iterable[MarkerDetection],
    ) -> Optional[ActiveTarget]:
        matches: List[tuple[float, PersonDetection, MarkerDetection]] = []
        for marker in markers:
            if not self._marker_allowed(marker.marker_id):
                continue
            for person in people:
                if point_in_bbox(marker.center, person.bbox):
                    px, py = person.center
                    mx, my = marker.center
                    matches.append((math.hypot(px - mx, py - my), person, marker))

        if not matches:
            return None

        _, person, marker = min(matches, key=lambda item: item[0])
        target = ActiveTarget(
            target_id=self._next_target_id,
            bbox=person.bbox,
            marker_id=marker.marker_id,
            source="marker",
        )
        self._next_target_id += 1
        return target

    def acquire_from_gesture(self, person: PersonDetection) -> ActiveTarget:
        target = ActiveTarget(
            target_id=self._next_target_id,
            bbox=person.bbox,
            marker_id=None,
            source="gesture",
        )
        self._next_target_id += 1
        return target

    def update_existing(
        self,
        target: ActiveTarget,
        people: Iterable[PersonDetection],
        frame_shape: tuple[int, int, int],
    ) -> Optional[ActiveTarget]:
        people_list = list(people)
        if not people_list:
            return None

        frame_h, frame_w = frame_shape[:2]
        center_threshold = max(frame_w, frame_h) * self.config.tracker_center_threshold_ratio

        best: Optional[tuple[float, float, PersonDetection]] = None
        for person in people_list:
            iou = bbox_iou(target.bbox, person.bbox)
            distance = center_distance(target.bbox, person.bbox)
            if iou >= self.config.tracker_iou_threshold or distance <= center_threshold:
                candidate = (iou, -distance, person)
                if best is None or candidate > best:
                    best = candidate

        if best is None:
            return None

        updated = ActiveTarget(
            target_id=target.target_id,
            bbox=best[2].bbox,
            marker_id=target.marker_id,
            source=target.source,
            missed_frames=0,
        )
        return updated

    def _marker_allowed(self, marker_id: int) -> bool:
        return self.config.target_marker_id is None or marker_id == self.config.target_marker_id
