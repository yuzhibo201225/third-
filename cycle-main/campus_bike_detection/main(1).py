from __future__ import annotations

import argparse
from pathlib import Path

from campus_bike_detection.models import CountLine, SystemConfig
from campus_bike_detection.system import BikeDetectionSystem

DEFAULT_MODEL = str(Path(__file__).parent / "yolov8n.pt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Campus Bike Real-time Detection")
    parser.add_argument("--source", default="0", help="camera id or video file path")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--backend", default="auto", choices=["auto", "pt", "onnx", "trt"])
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--imgsz", default=640, type=int)
    parser.add_argument("--conf", default=0.25, type=float)
    parser.add_argument("--iou", default=0.5, type=float)
    parser.add_argument("--line", default="0.05,0.5,0.95,0.5", help="count line in normalized coords: x1,y1,x2,y2")
    parser.add_argument("--count-direction", default="both", choices=["both", "forward", "backward"])
    parser.add_argument("--count-min-cross", default=0.003, type=float)
    parser.add_argument("--count-debounce-frames", default=5, type=int)
    parser.add_argument("--no-trails", action="store_true", help="disable track trails and side debug overlay")
    parser.add_argument("--no-show", action="store_true")
    return parser.parse_args()


def _parse_line(raw: str) -> CountLine:
    parts = [float(v.strip()) for v in raw.split(",")]
    if len(parts) != 4:
        raise ValueError("--line must be x1,y1,x2,y2")
    x1, y1, x2, y2 = parts
    for v in parts:
        if v < 0.0 or v > 1.0:
            raise ValueError("--line values must be normalized into [0,1]")
    return CountLine("main", (x1, y1), (x2, y2))


def main() -> None:
    args = parse_args()
    source: str | int = int(args.source) if args.source.isdigit() else args.source

    cfg = SystemConfig(
        source=source,
        model_path=args.model,
        backend=args.backend,
        device=args.device,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        show=not args.no_show,
        line=_parse_line(args.line),
        count_direction=args.count_direction,
        count_min_cross=args.count_min_cross,
        count_debounce_frames=args.count_debounce_frames,
        draw_trails=not args.no_trails,
    )

    with BikeDetectionSystem(cfg) as system:
        report = system.run()

    print("\n=== Session Report ===")
    print(f"Frames      : {report.total_frames}")
    print(f"Avg FPS     : {report.avg_fps:.2f}")
    print(f"Peak Count  : {report.peak_count}")
    print(f"Total Bikes : {report.total_count}")
    print(f"Line Counts : {report.line_counts}")


if __name__ == "__main__":
    main()