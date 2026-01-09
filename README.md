# Basketball Video Analysis System

GPU-accelerated video processing system for player detection, tracking, and court analysis in basketball videos.

## Demo

![Basketball Analysis Demo](asset/sample.gif)

## Features

- Automatic court keypoint detection
- Player segmentation and multi-object tracking
- Paint area density alert system
- Real-time minimap visualization
- Web-based user interface

## Technologies

### Backend
- NVIDIA DeepStream 7.1 (Triton)
- TensorRT 10.3.0
- CUDA 12.x (tensorrt-cu12-bindings)
- Python 3.x
- FastAPI
- OpenCV
- PyCUDA

### Frontend
- React 19.2
- TypeScript 5.9
- Vite 7.2 (rolldown-vite)
- Tailwind CSS 4.1

### Models
- YOLOv8 Pose Estimation (court detection)
- YOLOv8m-seg (player segmentation)

## Requirements

- Docker >= 20.10
  - [Installation Guide](https://docs.docker.com/engine/install/)
- Docker Compose >= 2.0
  - [Installation Guide](https://docs.docker.com/compose/install/)
- NVIDIA GPU (CUDA Compute Capability >= 7.5)
  - [Check GPU Compatibility](https://developer.nvidia.com/cuda-gpus)
- NVIDIA Container Toolkit >= 1.14.0
  - [Installation Guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
- NVIDIA Driver >= 525.60.13 (minimum for CUDA 12.x)
  - [Driver Downloads](https://www.nvidia.com/download/index.aspx)

## Installation and Running

```bash
# Clone the repository
git clone https://github.com/Sametcelikk/basketball_case.git
cd basketball_case

# IMPORTANT: Before building, copy ONNX models from the provided zip file to /models directory
# Required files:
#   - custom-pose-model-2.onnx
#   - yolov8m-seg.onnx

# Build and start containers
docker compose up --build

# Or separately:
# docker compose build
# docker compose up

```

On first run, TensorRT engine files will be automatically generated from ONNX models (may take 5-10 minutes).

**Access:**
- Frontend: http://localhost:5173 or http://127.0.0.1:5173

## Usage

1. Open `http://localhost:5173` or `http://127.0.0.1:5173` in your browser
2. Select one of the available videos
3. Click "Process Video" button
4. Watch or download the video when processing is complete

## Project Structure

```
basketball_case/
├── src/                          # Backend source code
│   ├── main.py                   # Video processing pipeline
│   ├── local.py                  # FastAPI web server
│   ├── deepstream_inference.py   # TensorRT model wrapper
│   ├── court_utils.py            # Court analysis
│   ├── segmentation.py           # Player segmentation
│   ├── tracker.py                # Multi-object tracking
│   ├── minimap.py                # Minimap generation
│   └── alert.py                  # Paint area alert system
├── frontend/                     # React frontend
├── models/                       # ONNX model files
├── deepstream_configs/           # Model configurations
├── static/
│   ├── videos/                   # Input videos
│   └── outputs/                  # Processed videos
├── docker-compose.yml
├── Dockerfile
└── requirements.txt
```
