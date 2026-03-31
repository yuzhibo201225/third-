# Campus Bike Real-time Detection

精简后的工程仅保留核心链路：**检测 → 追踪 → 计数 → FPS 显示**，并支持 **PyTorch / ONNX / TensorRT** 推理。

## Re-structured Project

```text
campus_bike_detection/
  main.py          # CLI 入口
  system.py        # 主循环与可视化
  detector.py      # 多后端推理
  tracker.py       # 轻量追踪
  flow_counter.py  # 跨线计数
  models.py        # 数据结构
scripts/
  export_onnx.py
  build_tensorrt.py
  infer_onnx.py
  infer_trt.py
docs/
  EDGE_DEPLOYMENT.md
```

## Quick Start

```bash
pip install -r requirements.txt
python -m campus_bike_detection.main --source 0 --model campus_bike_detection/yolov8n.pt --backend pt --device cuda
```

## ONNX / TensorRT

```bash
python scripts/export_onnx.py --model campus_bike_detection/yolov8n.pt --imgsz 640
python scripts/build_tensorrt.py --model campus_bike_detection/yolov8n.pt --imgsz 640 --half
```

```bash
python -m campus_bike_detection.main --source 0 --model yolov8n.onnx --backend onnx --device cpu
python -m campus_bike_detection.main --source 0 --model yolov8n.engine --backend trt --device cuda
```

## Lightweight Model Suggestions
- First choice: `yolov8n`.
- For edge throughput: TensorRT FP16 / INT8.
- For CPU-only board: ONNX + `--imgsz 320` + `--no-show`.
