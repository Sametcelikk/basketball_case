# syntax=docker/dockerfile:1
FROM nvcr.io/nvidia/deepstream:7.1-triton-multiarch

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y \
    python3-pip \
    python3-dev \
    python3-gi \
    python3-gst-1.0 \
    libgomp1 \
    libgstreamer1.0-0 \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    gstreamer1.0-libav \
    gstreamer1.0-tools \
    wget \
    curl \
    xz-utils

# FFmpeg
RUN wget -q https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz && \
    tar xf ffmpeg-release-amd64-static.tar.xz && \
    mv ffmpeg-*-amd64-static/ffmpeg /usr/local/bin/ && \
    mv ffmpeg-*-amd64-static/ffprobe /usr/local/bin/ && \
    rm -rf ffmpeg-* && \
    chmod +x /usr/local/bin/ffmpeg /usr/local/bin/ffprobe

# TensorRT
RUN --mount=type=cache,target=/root/.cache/pip \
    pip3 install --upgrade pip wheel && \
    pip3 cache remove "tensorrt*" || true && \
    pip3 install --extra-index-url https://pypi.nvidia.com \
        tensorrt==10.3.0 \
        tensorrt-cu12-bindings==10.3.0 \
        pycuda

# DeepStream Python bindings
RUN wget https://github.com/NVIDIA-AI-IOT/deepstream_python_apps/releases/download/v1.1.11/pyds-1.1.11-py3-none-linux_x86_64.whl -O /tmp/pyds-1.1.11-py3-none-linux_x86_64.whl && \
    pip3 install /tmp/pyds-1.1.11-py3-none-linux_x86_64.whl && \
    rm /tmp/pyds-1.1.11-py3-none-linux_x86_64.whl

WORKDIR /app

COPY requirements.txt .

RUN --mount=type=cache,target=/root/.cache/pip \
    pip3 install -r requirements.txt

COPY deepstream_configs/ ./deepstream_configs/
COPY static/ ./static/
COPY models/ ./models/
COPY src/ ./src/
COPY engine_check/ ./engine_check/

RUN mkdir -p /app/static/videos /app/static/outputs

RUN chmod +x /app/engine_check/*.py /app/engine_check/*.sh && \
    sed -i 's/\r$//' /app/engine_check/*.sh

EXPOSE 8000

ENV USE_DEEPSTREAM=true
ENV GST_PLUGIN_PATH=/opt/nvidia/deepstream/deepstream/lib/gst-plugins/

ENV CUDA_MODULE_LOADING=LAZY

ENTRYPOINT ["/app/engine_check/entrypoint.sh"]
CMD ["python3", "src/local.py"]
