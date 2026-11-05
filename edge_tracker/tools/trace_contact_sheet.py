#!/usr/bin/env python3
"""Draw recorded boxes, track IDs and ReID scores onto frames of a recorded run.

Writes one image with evenly spaced frames between two times (seconds from the
start of the trace). The speaker's box is yellow, other people green, markers red.

    python tools/trace_contact_sheet.py room_01 41 47 12 sheet.jpg
"""
from __future__ import annotations

import json
import sys

import cv2
import numpy as np


def main() -> None:
    if len(sys.argv) != 6:
        raise SystemExit(__doc__)
    name, time_from, time_to, count, output = sys.argv[1:]
    time_from, time_to, count = float(time_from), float(time_to), int(count)
    with open(f"test_video/traces/{name}.jsonl", encoding="utf-8") as stream:
        records = [json.loads(line) for line in stream]
    capture = cv2.VideoCapture(f"test_video/raw/{name}.mp4")
    started_at = records[0]["timestamp"]
    # Video frame i and trace line i are the same processed frame.
    indexes = [
        index
        for index, record in enumerate(records)
        if time_from <= record["timestamp"] - started_at <= time_to
    ]
    picked = [indexes[int(k)] for k in np.linspace(0, len(indexes) - 1, count)]
    tiles = []
    for index in picked:
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = capture.read()
        if not ok:
            continue
        record = records[index]
        for item in record["detections"]:
            x1, y1, x2, y2 = (int(value) for value in item["box"])
            color = (0, 255, 0) if item["label"] == "person" else (0, 0, 255)
            if item["track_id"] == record["decision"]["target_track_id"]:
                color = (0, 255, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            label = str(item["track_id"])
            if item["appearance_similarity"] is not None:
                label += f" {item['appearance_similarity']:.2f}"
            cv2.putText(frame, label, (x1 + 2, y1 + 16), 0, 0.55, color, 2)
        caption = f"{record['timestamp'] - started_at:.2f}s {record['decision']['state'][:10]}"
        cv2.putText(frame, caption, (4, frame.shape[0] - 6), 0, 0.55, (255, 255, 255), 2)
        tiles.append(frame)
    columns = 3
    rows = [
        np.hstack(tiles[start : start + columns])
        for start in range(0, len(tiles) - len(tiles) % columns, columns)
    ]
    cv2.imwrite(output, np.vstack(rows))
    print("wrote", output, "frames", len(rows) * columns)


if __name__ == "__main__":
    main()
