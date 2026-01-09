#!/usr/bin/env python3

import os
import subprocess
import sys

if __name__ == "__main__":
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else '/app/engine_check'
    PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
else:
    PROJECT_DIR = '/app'

MODELS_DIR = os.path.join(PROJECT_DIR, "models")

ONNX_MODELS = [
    {
        "name": "Pose Model",
        "onnx": os.path.join(MODELS_DIR, "custom-pose-model-2.onnx"),
        "engine": os.path.join(MODELS_DIR, "custom-pose-model-2.engine"),
    },
    {
        "name": "Segmentation Model",
        "onnx": os.path.join(MODELS_DIR, "yolov8m-seg.onnx"),
        "engine": os.path.join(MODELS_DIR, "yolov8m-seg.engine"),
    },
]


def convert_onnx_to_engine(onnx_path, engine_path, fp16=True, workspace_mb=2048):
    if not os.path.exists(onnx_path):
        print(f"ONNX model not found: {onnx_path}")
        return False

    if os.path.exists(engine_path):
        print(f"Engine already exists: {engine_path}")

        engine_size = os.path.getsize(engine_path)
        if engine_size == 0:
            print(f"   Corrupted engine (0 bytes), rebuilding...")
            os.remove(engine_path)
        else:
            engine_size_mb = engine_size / (1024 * 1024)
            print(f"   Engine size: {engine_size_mb:.2f} MB")
            print(f"   Skipping conversion")
            return True

    print(f"\nConverting: {os.path.basename(onnx_path)}")
    print(f"   -> Engine: {os.path.basename(engine_path)}")

    cmd = [
        "trtexec",
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        f"--memPoolSize=workspace:{workspace_mb}MiB",
    ]

    if fp16:
        cmd.append("--fp16")

    print(f"   Parameters:")
    print(f"   - FP16: {fp16}")
    print(f"   - Workspace: {workspace_mb} MiB")

    try:
        print(f"\n   Conversion started (this may take a few minutes)...")
        sys.stdout.flush()
        
        result = subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        
        print(f"   Conversion completed: {os.path.basename(engine_path)}")

        engine_size_mb = os.path.getsize(engine_path) / (1024 * 1024)
        print(f"   Engine size: {engine_size_mb:.2f} MB")

        return True
    except subprocess.CalledProcessError as e:
        print(f"\n   Conversion failed:")
        if e.stderr:
            print(e.stderr)
        return False


def main():
    print("=" * 60)
    print("TensorRT Engine Conversion")
    print("=" * 60)
    
    print(f"SCRIPT_DIR: {SCRIPT_DIR}")
    print(f"PROJECT_DIR: {PROJECT_DIR}")
    print(f"MODELS_DIR: {MODELS_DIR}")
    
    try:
        subprocess.run(["trtexec", "--help"], capture_output=True, check=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("trtexec not found!")
        print("This script must be run inside DeepStream container.")
        sys.exit(1)
    
    if not os.path.exists(MODELS_DIR):
        print(f"Models folder not found: {MODELS_DIR}")
        sys.exit(1)
    
    print(f"Models folder found: {MODELS_DIR}")
    
    for model_info in ONNX_MODELS:
        if os.path.exists(model_info["onnx"]):
            print(f"{model_info['name']}: {model_info['onnx']}")
        else:
            print(f"{model_info['name']}: {model_info['onnx']} NOT FOUND!")
    
    print("")
    
    success_count = 0
    total_count = len(ONNX_MODELS)
    
    for model_info in ONNX_MODELS:
        print(f"\n{model_info['name']}")
        if convert_onnx_to_engine(model_info["onnx"], model_info["engine"]):
            success_count += 1
    
    print("\n" + "=" * 60)
    print(f"Completed: {success_count}/{total_count} models converted successfully")
    print("=" * 60)
    
    if success_count < total_count:
        sys.exit(1)


if __name__ == "__main__":
    main()

