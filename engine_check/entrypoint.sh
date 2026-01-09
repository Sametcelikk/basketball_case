#!/bin/bash

set -e

echo "=========================================="
echo "DeepStream Basketball Analysis"
echo "=========================================="

MODELS_DIR="/app/models"
POSE_ENGINE="${MODELS_DIR}/custom-pose-model-2.engine"
SEG_ENGINE="${MODELS_DIR}/yolov8m-seg.engine"

POSE_ONNX="${MODELS_DIR}/custom-pose-model-2.onnx"
SEG_ONNX="${MODELS_DIR}/yolov8m-seg.onnx"

echo ""
echo "Checking model files..."

ENGINE_MISSING=false

if [ ! -f "$POSE_ENGINE" ]; then
    echo "Pose engine not found: $POSE_ENGINE"
    ENGINE_MISSING=true
fi

if [ ! -f "$SEG_ENGINE" ]; then
    echo "Segmentation engine not found: $SEG_ENGINE"
    ENGINE_MISSING=true
fi

if [ "$ENGINE_MISSING" = true ]; then
    echo ""
    echo "Engine files missing, converting from ONNX..."
    echo "This may take 5-10 minutes on first run"
    echo ""

    python3 /app/engine_check/convert_to_engine.py

    if [ $? -eq 0 ]; then
        echo ""
        echo "Engine conversion completed successfully"
        
        if [ ! -f "$POSE_ENGINE" ] || [ ! -f "$SEG_ENGINE" ]; then
            echo "ERROR: Engine files not found after conversion"
            exit 1
        fi
        
        POSE_SIZE=$(stat -c%s "$POSE_ENGINE" 2>/dev/null || stat -f%z "$POSE_ENGINE" 2>/dev/null || echo "0")
        SEG_SIZE=$(stat -c%s "$SEG_ENGINE" 2>/dev/null || stat -f%z "$SEG_ENGINE" 2>/dev/null || echo "0")
        
        if [ "$POSE_SIZE" -eq 0 ] || [ "$SEG_SIZE" -eq 0 ]; then
            echo "ERROR: Engine files are corrupted"
            exit 1
        fi
        
        echo "Pose engine: $(basename $POSE_ENGINE)"
        echo "Segmentation engine: $(basename $SEG_ENGINE)"
    else
        echo ""
        echo "Engine conversion failed"
        echo "Check if ONNX files exist:"
        echo "- $POSE_ONNX"
        echo "- $SEG_ONNX"
        exit 1
    fi
else
    echo "All engine files found"
    echo "- Pose engine: $(basename $POSE_ENGINE)"
    echo "- Segmentation engine: $(basename $SEG_ENGINE)"
fi

echo ""
echo "=========================================="
echo "Starting web server..."
echo "=========================================="
echo ""

exec "$@"
