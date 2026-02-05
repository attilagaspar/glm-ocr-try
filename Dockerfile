# Dockerfile for GLM-OCR with Ollama and GPU support
FROM ollama/ollama:0.15.5-rc2

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    curl \
    wget \
    git \
    zstd \
    poppler-utils \
    tesseract-ocr \
    libtesseract-dev \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Install Python dependencies
RUN pip3 install --no-cache-dir --break-system-packages \
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
RUN pip3 install --no-cache-dir --break-system-packages \
    rapidocr-onnxruntime \
    modelscope \
    easyocr

# Create directories for models and data
RUN mkdir -p /workspace/models /workspace/data /workspace/output

# Expose Ollama port
EXPOSE 11434

# Copy application files
COPY . /workspace/

# Set up entrypoint - keep container running
CMD ["tail", "-f", "/dev/null"]
