# GOLDEN_DOME_AI — Real‑Time Defense Detection Module

## Overview
The **GOLDEN_DOME_AI** subsystem is a high‑performance real‑time object‑tracking and detection engine optimized for:
- **GPU acceleration** (NVIDIA CUDA 11+)
- **Parallel video processing**
- **Ultra‑low‑latency YOLO inference**
- **Modular integration** with external hardware controllers (e.g., ESP32, servo systems)

This module is designed to act as a **library-style callable detection system**, allowing other components in the Iron Dome prototype to easily access object center‑pixel coordinates in real time.

---

## Repository Structure
```
GOLDEN_DOME_AI/
│── shared_data.py         # Thread-safe shared memory between detector and hardware
│── realtime_yolo.py       # Main high-speed video inference pipeline
│── yolov11m.pt            # Model weights (user provided)
│── requirements.txt       # All dependencies (GPU + CPU fallback)
│── utils/
│    └── camera_manager.py # Camera abstraction layer
│
└── README.md              # This file
```

---

## Installation

### 1. Create virtual environment
```bash
python -m venv venv_stable
venv_stable\Scripts\activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Verify CUDA availability
```python
import torch
print(torch.cuda.is_available())
print(torch.version.cuda)
```

---

## Running the System
Run real‑time detection:
```bash
python realtime_yolo.py
```

---

## Using This Module as a Library

You can import the detection loop or access shared output variables from **shared_data.py**.

### Example — Retrieve object pixel center from another Python script
```python
from shared_data import SharedState

state = SharedState()

# Get latest center coordinate (thread-safe)
x, y = state.get_center()

print("Object Center:", x, y)
```

### Example — Subscribe to detection events
```python
from shared_data import SharedState

s = SharedState()

if s.object_detected:
    print("Target Acquired:", s.get_center())
```

### Example — Reset the detector state
```python
s = SharedState()
s.reset()
```

---

## Output Variables Exposed by the Library

| Variable | Type | Description |
|---------|------|-------------|
| `state.center_x` | `int` | X‑coordinate of object center |
| `state.center_y` | `int` | Y‑coordinate of object center |
| `state.object_detected` | `bool` | True only when target class is detected |
| `state.last_frame` | `numpy array` | Raw last processed frame (optional use) |

All variables inside **SharedState** are **thread‑safe** and can be safely called by:
- Motor controllers  
- Web dashboards  
- External clients  
- Event‑loop‑based systems  

---

## GPU Fallback Logic

If CUDA is available:
```
Using device: cuda
```
Otherwise:
```
Using device: cpu
```

Automatically handled inside realtime_yolo.py.

---

## Notes
- Ensure your NVIDIA driver + CUDA version is correctly installed (CUDA 12+ supported)
- Camera index defaults to `0`, modify in config section if needed
- YOLOv11m model will automatically optimize itself on first run

---

## License
This module is part of the closed prototype **Golden Dome AI Defense Project** and is not open‑source.

