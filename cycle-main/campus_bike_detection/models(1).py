from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class Detection:
    bbox: tuple[float, float, float, float]  # x1,y1,x2,y2 normalized
    confidence: float
    class_id: int


@dataclass(slots=True)
class Track:
    track_id: int
    bbox: tuple[float, float, float, float]
    confidence: float
    trajectory: list[tuple[float, float]] = field(default_factory=list)


@dataclass(slots=True)
class CountLine:
    line_id: str
    start: tuple[float, float]
    end: tuple[float, float]


@dataclass(slots=True)
class SessionReport:
    total_frames: int
    avg_fps: float
    peak_count: int
    total_count: int
    line_counts: dict[str, int]


@dataclass(slots=True)
class SystemConfig:
    source: str | int
    model_path: str
    backend: str = "auto"  # auto|pt|onnx|trt
    device: str = "cuda"  # cuda|cpu
    conf: float = 0.25
    iou: float = 0.5
    imgsz: int = 640
    show: bool = True
    line: CountLine = field(default_factory=lambda: CountLine("main", (0.05, 0.5), (0.95, 0.5)))
    count_direction: str = "both"  # both|forward|backward
    count_min_cross: float = 0.003
    count_debounce_frames: int = 5
    draw_trails: bool = True