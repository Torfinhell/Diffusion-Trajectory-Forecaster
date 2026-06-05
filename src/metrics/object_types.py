"""Waymo / Waymax ObjectTypeIds (WOMD convention)."""

from typing import Mapping

# 0: UNSET/UNKNOWN, 1: VEHICLE, 2: PEDESTRIAN, 3: CYCLIST
OBJECT_TYPE_UNKNOWN = 0
OBJECT_TYPE_VEHICLE = 1
OBJECT_TYPE_PEDESTRIAN = 2
OBJECT_TYPE_CYCLIST = 3

DEFAULT_TYPE_LABELS: Mapping[int, str] = {
    0: "unknown",
    1: "vehicle",
    2: "pedestrian",
    3: "cyclist",
}
