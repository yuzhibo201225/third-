from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from campus_bike_detection.models import Detection

BICYCLE_CLASS_ID = 1


class BikeDetector:
    def __init__(
        self,
        model_path: str,
        backend: str = "auto",
        device: str = "cuda",
        conf: float = 0.25,
        iou: float = 0.5,
        imgsz: int = 640,
    ) -> None:
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(model_path)

        self.device = device
        self.conf = conf
        self.iou = iou
        self.imgsz = imgsz
        self.backend = self._resolve_backend(backend)
        self.model = self._load_model()

    def _resolve_backend(self, backend: str) -> str:
        if backend != "auto":
            return backend
        ext = self.model_path.suffix.lower()
        if ext == ".pt":
            return "pt"
        if ext == ".onnx":
            return "onnx"
        if ext in {".engine", ".trt"}:
            return "trt"
        raise ValueError(f"Unsupported model extension: {ext}")

    def _load_model(self):
        if self.backend == "pt":
            from ultralytics import YOLO

            model = YOLO(str(self.model_path))
            return model

        if self.backend == "onnx":
            import onnxruntime as ort

            providers = ["CPUExecutionProvider"]
            if self.device == "cuda":
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            return ort.InferenceSession(str(self.model_path), providers=providers)

        if self.backend == "trt":
            from ultralytics import YOLO

            return YOLO(str(self.model_path))

        raise ValueError(f"Unsupported backend: {self.backend}")

    def warmup(self) -> None:
        frame = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
        _ = self.detect(frame)

    def detect(self, frame: np.ndarray) -> list[Detection]:
        if self.backend in {"pt", "trt"}:
            return self._detect_yolo(frame)
        return self._detect_onnx(frame)

    def _detect_yolo(self, frame: np.ndarray) -> list[Detection]:
        results = self.model(
            frame,
            classes=[BICYCLE_CLASS_ID],
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            verbose=False,
            device=self.device,
        )
        return self._from_ultralytics(results)

    def _from_ultralytics(self, results) -> list[Detection]:
        detections: list[Detection] = []
        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                cls = int(box.cls[0].item())
                if cls != BICYCLE_CLASS_ID:
                    continue
                x1, y1, x2, y2 = [float(v) for v in box.xyxyn[0].tolist()]
                detections.append(
                    Detection(
                        bbox=(x1, y1, x2, y2),
                        confidence=float(box.conf[0].item()),
                        class_id=cls,
                    )
                )
        return detections

    def _detect_onnx(self, frame: np.ndarray) -> list[Detection]:
        h, w = frame.shape[:2]
        resized = cv2.resize(frame, (self.imgsz, self.imgsz))
        inp = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        inp = np.transpose(inp, (2, 0, 1))[None, ...]

        out = self.model.run(None, {self.model.get_inputs()[0].name: inp})[0]
        rows = out[0] if out.ndim == 3 else out
        if rows.shape[0] == 6:
            rows = rows.T

        detections: list[Detection] = []
        for row in rows:
            if len(row) < 6:
                continue
            x1, y1, x2, y2, conf, cls = [float(v) for v in row[:6]]
            if int(cls) != BICYCLE_CLASS_ID or conf < self.conf:
                continue
            detections.append(
                Detection(
                    bbox=(max(0.0, x1 / w), max(0.0, y1 / h), min(1.0, x2 / w), min(1.0, y2 / h)),
                    confidence=conf,
                    class_id=int(cls),
                )
            )
        return detections
