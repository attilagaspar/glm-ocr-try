# GLM-OCR Docker Environment

This project provides a Docker environment for running GLM-4V OCR with GPU support to extract tables from images and PDFs.

## Prerequisites

- Linux system with NVIDIA GPU
- Docker installed
- Docker Compose installed
- NVIDIA Container Toolkit (nvidia-docker2) installed
- NVIDIA GPU drivers

### Installing NVIDIA Container Toolkit

```bash
# Ubuntu/Debian
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

## Quick Start

1. **Make the setup script executable:**
   ```bash
   chmod +x setup.sh
   ```

2. **Run the setup script:**
   ```bash
   ./setup.sh
   ```
   This will:
   - Build the Docker image with all dependencies
   - Start the container
   - Pull the GLM-4V model via Ollama

3. **Process your files:**
   ```bash
   # Place your JPG/PDF files in the example-data/ folder
   docker-compose exec glm-ocr python3 ocr_to_table.py
   ```

4. **Check the results:**
   - Results will be saved in the `output/` folder
   - Tables are exported as Excel (.xlsx) files
   - JSON summaries are also generated

## Manual Setup

If you prefer to set up manually:

```bash
# Build the image
docker-compose build

# Start the container
docker-compose up -d

# Start Ollama and pull the model
docker-compose exec glm-ocr bash -c "ollama serve &"
docker-compose exec glm-ocr ollama pull glm4v:9b

# Run the OCR script
docker-compose exec glm-ocr python3 ocr_to_table.py
```

## Directory Structure

```
.
├── Dockerfile              # Docker image definition
├── docker-compose.yml      # Docker Compose configuration
├── setup.sh               # Setup script
├── ocr_to_table.py        # Main Python OCR script
├── example-data/          # Input files (JPG/PDF)
├── output/                # Generated results
└── models/                # Cached models
```

## Usage

### Processing Files

1. Place your JPG or PDF files in the `example-data/` directory
2. Run the processing script:
   ```bash
   docker-compose exec glm-ocr python3 ocr_to_table.py
   ```

### Output Formats

The script generates:
- **Excel files** (.xlsx): One file per page with all tables as separate sheets
- **JSON files**: Complete extraction results with metadata
- **Text files**: Raw model responses (if parsing fails)

### Custom Processing

You can also use the Python class directly:

```python
from ocr_to_table import GLMOCRTableExtractor

# Initialize
extractor = GLMOCRTableExtractor(model_name="glm4v:9b")

# Process a single file
extractor.process_file("/workspace/data/document.pdf", output_format="excel")

# Or extract from an image
table_data = extractor.extract_table_with_glm("/workspace/data/table.jpg")
```

## GPU Configuration

The Docker Compose file is configured to use all available GPUs. To use specific GPUs:

```yaml
# In docker-compose.yml, modify:
environment:
  - NVIDIA_VISIBLE_DEVICES=0,1  # Use only GPU 0 and 1
```

## Troubleshooting

### Ollama Connection Issues

If you get connection errors:
```bash
# Check if Ollama is running
docker-compose exec glm-ocr pgrep -f "ollama serve"

# Start Ollama manually
docker-compose exec glm-ocr bash -c "ollama serve &"
```

### GPU Not Detected

```bash
# Test GPU access
docker-compose exec glm-ocr nvidia-smi
```

### Model Not Found

```bash
# Pull the model manually
docker-compose exec glm-ocr ollama pull glm4v:9b
```

## Stopping the Environment

```bash
# Stop the container
docker-compose down

# Stop and remove volumes
docker-compose down -v
```

## Notes

- The GLM-4V 9B model requires significant GPU memory (at least 16GB VRAM recommended)
- First run will take longer as the model needs to be downloaded
- PDF processing converts each page to an image first
- Results quality depends on image quality and table complexity
