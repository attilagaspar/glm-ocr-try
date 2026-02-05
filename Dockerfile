# Dockerfile for GLM-OCR with Ollama and GPU support
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    curl \
    wget \
    git \
    poppler-utils \
    tesseract-ocr \
    libtesseract-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# Set working directory
WORKDIR /workspace

# Install Python dependencies
RUN pip3 install --no-cache-dir \
    torch \
    torchvision \
    pillow \
    pdf2image \
    opencv-python \
    numpy \
    pandas \
    openpyxl \
    transformers \
    accelerate \
    sentencepiece \
    protobuf

# Install GLM-OCR dependencies
RUN pip3 install --no-cache-dir \
    rapidocr-onnxruntime \
    modelscope \
    easyocr

# Create directories for models and data
RUN mkdir -p /workspace/models /workspace/data /workspace/output

# Expose Ollama port
EXPOSE 11434

# Copy application files
COPY . /workspace/

# Set up entrypoint
CMD ["/bin/bash"]
