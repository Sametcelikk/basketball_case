import sys
import uuid
import shutil
from pathlib import Path

from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import subprocess
import uvicorn

SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
STATIC_DIR = PROJECT_DIR / "static"
VIDEOS_DIR = STATIC_DIR / "videos"
OUTPUT_DIR = STATIC_DIR / "outputs"

VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Basketball DeepStream API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5173",
        "http://localhost:5173"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

processing_status: dict = {}


@app.get("/")
async def home():
    return {"message": "Basketball DeepStream API", "status": "ok"}


@app.get("/gpu-check")
async def gpu_check():
    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
        if result.returncode == 0:
            return {
                "status": "ok",
                "message": "GPU access successful",
                "output": result.stdout,
            }
        else:
            return {
                "status": "error",
                "message": "GPU not found",
                "error": result.stderr,
            }
    except Exception as e:
        return {"status": "error", "message": str(e)}


@app.get("/available-videos")
async def get_available_videos():
    videos = []

    for video_file in VIDEOS_DIR.glob("*.mp4"):
        videos.append({
            "name": video_file.stem,
            "filename": video_file.name,
            "path": f"/static/videos/{video_file.name}"
        })

    return {
        "status": "ok",
        "videos": sorted(videos, key=lambda x: x['name'])
    }


def run_video_processing(job_id: str, input_path: Path, output_path: Path):
    try:
        processing_status[job_id] = {
            "status": "processing",
            "progress": 0,
            "message": "Initializing...",
            "stage": "initializing"
        }

        sys.path.insert(0, str(SCRIPT_DIR))

        from main import process_video

        def progress_callback(current: int, total: int):
            percent = int((current / total) * 100) if total > 0 else 0

            if percent < 10:
                stage = "pose_detection"
                stage_text = "Detecting court boundaries"
            elif percent < 30:
                stage = "tracking"
                stage_text = "Tracking players"
            elif percent < 60:
                stage = "segmentation"
                stage_text = "Segmenting players"
            elif percent < 80:
                stage = "minimap"
                stage_text = "Generating minimap"
            else:
                stage = "finalizing"
                stage_text = "Finalizing processing"

            processing_status[job_id] = {
                "status": "processing",
                "progress": percent,
                "message": stage_text,
                "stage": stage
            }

        process_video(str(input_path), str(output_path), progress_callback)

        processing_status[job_id] = {
            "status": "processing",
            "progress": 95,
            "message": "Optimizing video",
            "stage": "export"
        }

        temp_output = output_path.parent / f"{output_path.stem}_h264.mp4"

        ffmpeg_cmd = [
            "ffmpeg", "-i", str(output_path),
            "-c:v", "libx264",
            "-preset", "veryfast",
            "-crf", "22",
            "-movflags", "+faststart",
            "-y",
            str(temp_output)
        ]

        result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)

        if result.returncode == 0:
            output_path.unlink()
            shutil.move(str(temp_output), str(output_path))
            print(f"Video converted to H.264 (FFmpeg): {output_path.name}")
        else:
            print(f"FFmpeg error: {result.stderr}")
            if temp_output.exists():
                temp_output.unlink()
            raise Exception(f"FFmpeg conversion error: {result.stderr[:200]}")

        processing_status[job_id] = {
            "status": "completed",
            "progress": 100,
            "message": "Completed!",
            "stage": "completed",
            "output_url": f"/static/outputs/{output_path.name}"
        }

    except Exception as e:
        processing_status[job_id] = {
            "status": "error",
            "progress": 0,
            "message": f"Error: {str(e)}"
        }


@app.post("/process")
async def process_video_endpoint(video_name: str, background_tasks: BackgroundTasks):
    for job_id, status in processing_status.items():
        if status.get("status") in ["starting", "processing"]:
            return {
                "status": "error",
                "message": "Another video is currently being processed. Please wait for it to complete."
            }

    input_path = VIDEOS_DIR / f"{video_name}.mp4"

    if not input_path.exists():
        return {
            "status": "error",
            "message": f"Video not found: {video_name}.mp4"
        }

    job_id = str(uuid.uuid4())[:8]
    output_filename = f"{job_id}_output.mp4"
    output_path = OUTPUT_DIR / output_filename

    processing_status[job_id] = {
        "status": "starting",
        "progress": 0,
        "message": "Starting process...",
        "video_name": video_name
    }

    background_tasks.add_task(run_video_processing, job_id, input_path, output_path)

    return {
        "status": "ok",
        "job_id": job_id,
        "video_name": video_name,
        "message": "Processing started"
    }


@app.get("/status/{job_id}")
async def get_status(job_id: str):
    if job_id not in processing_status:
        return {
            "status": "unknown",
            "message": "Job not found"
        }

    return processing_status[job_id]


@app.get("/video/{job_id}")
async def get_video(job_id: str):
    output_files = list(OUTPUT_DIR.glob(f"{job_id}_output.*"))

    if not output_files:
        return {"status": "error", "message": "Video not found"}

    return FileResponse(
        output_files[0],
        media_type="video/mp4",
        filename=f"processed_{job_id}.mp4"
    )


@app.get("/videos")
async def list_videos():
    videos = []

    for output_file in OUTPUT_DIR.glob("*_output.mp4"):
        job_id = output_file.name.split("_")[0]
        stat = output_file.stat()
        videos.append({
            "job_id": job_id,
            "filename": output_file.name,
            "url": f"/static/outputs/{output_file.name}",
            "timestamp": stat.st_mtime
        })

    videos.sort(key=lambda x: x["timestamp"], reverse=True)
    return {"videos": videos}


if __name__ == "__main__":
    print("Checking GPU access...")

    result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
    if result.returncode == 0:
        print("GPU access OK")
        print(result.stdout)
    else:
        print("GPU not found!")
        print(result.stderr)

    print("\n=== Web Server Starting ===")
    print("Interface: http://127.0.0.1:5173")
    uvicorn.run("local:app", host="0.0.0.0", port=8000, reload=True)
