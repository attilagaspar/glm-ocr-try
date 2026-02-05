#!/bin/bash
# Setup script for GLM-OCR Docker environment

set -e

echo "=== GLM-OCR Docker Setup Script ==="
echo ""

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "Error: Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "Error: Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Check for NVIDIA Docker runtime
if ! docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi &> /dev/null; then
    echo "Error: NVIDIA Docker runtime is not properly configured."
    echo "Please install nvidia-docker2 and configure the NVIDIA Container Toolkit."
    exit 1
fi

echo "✓ Docker and NVIDIA runtime are properly configured"
echo ""

# Create necessary directories
echo "Creating directories..."
mkdir -p output models

echo "✓ Directories created"
echo ""

# Build Docker image
echo "Building Docker image (this may take several minutes)..."
docker-compose build

echo "✓ Docker image built successfully"
echo ""

# Start the container
echo "Starting container..."
docker-compose up -d

echo "✓ Container started"
echo ""

# Wait for Ollama to be ready and pull the model
echo "Starting Ollama service and pulling GLM4V model..."
docker-compose exec glm-ocr bash -c "ollama serve &" || true
sleep 5

echo "Pulling glm4v-9b model (this will take some time)..."
docker-compose exec glm-ocr ollama pull glm4v:9b || echo "Note: Model pull may need to be done manually"

echo ""
echo "=== Setup Complete ==="
echo ""
echo "To run the OCR script:"
echo "  docker-compose exec glm-ocr python3 ocr_to_table.py"
echo ""
echo "To access the container shell:"
echo "  docker-compose exec glm-ocr bash"
echo ""
echo "To stop the container:"
echo "  docker-compose down"
