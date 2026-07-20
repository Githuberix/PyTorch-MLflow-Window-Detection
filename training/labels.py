from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "sample"
TIME_FORMAT = "%Y-%m-%d %H:%M:%S"
ROOMS = ("BZ", "EZ", "SZ", "WZ")
INPUT_FILE = DATA_DIR / "fused_temperature_features.csv"
OUTPUT_FILE = DATA_DIR / "labeled_temperature_features.csv"

WINDOW_EVENTS = {
    "BZ": [
        ("2026-05-03 16:05:00", "2026-05-03 16:40:00"),
        ("2026-05-03 20:00:00", "2026-05-03 22:50:00"),
        ("2026-05-04 08:15:00", "2026-05-04 08:55:00"),
    ],
    "EZ": [],
    "SZ": [],
    "WZ": [],
}


def parse_time(value: str) -> datetime:
    return datetime.strptime(value, TIME_FORMAT)


def main() -> None:
    ranges = {
        room: [(parse_time(start), parse_time(end)) for start, end in room_ranges]
        for room, room_ranges in WINDOW_EVENTS.items()
    }

    with INPUT_FILE.open(newline="", encoding="utf-8-sig") as file:
        rows = list(csv.DictReader(file))

    labeled_rows = []
    for row in rows:
        timestamp = parse_time(row["time"])
        labels = []
        labeled_row = dict(row)
        for room in ROOMS:
            label = int(any(start <= timestamp <= end for start, end in ranges[room]))
            labeled_row[f"label_window_{room}"] = label
            labels.append(label)
        labeled_row["label_any_window"] = int(any(labels))
        labeled_rows.append(labeled_row)

    with OUTPUT_FILE.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(labeled_rows[0].keys()))
        writer.writeheader()
        writer.writerows(labeled_rows)

    print(f"Wrote {len(labeled_rows)} rows to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
