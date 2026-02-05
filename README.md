# GLM-OCR Docker Environment

This project provides a Docker environment for running GLM-4V OCR with GPU support to extract tables from images and PDFs.

## Prerequisites

**These should already be installed by your system administrator:**
- Linux system with NVIDIA GPU
- Docker (with user access to run docker commands)
- Docker Compose
- NVIDIA Container Toolkit (nvidia-docker2)
- NVIDIA GPU drivers

**No sudo/admin access required** - all setup runs within Docker containers.

## Quick Start

1. **Create directories for your data (on the Linux host):**
   ```bash
   mkdir -p /home/gaspar/econai/data
   mkdir -p /home/gaspar/econai/output
   ```

2. **Make the setup script executable:**
   ```bash
   chmod +x setup.sh
   ```

3. **Run the setup script:**
   ```bash
   ./setup.sh
   ```
   This will:
   - Build the Docker image with all dependencies
   - Start the container
   - Pull the GLM-4V model via Ollama

4. **Place your JPG/PDF files:**
   ```bash
   # On the host, copy files to:
   cp your_file.jpg /home/gaspar/econai/data/
   # Inside container, they'll appear at: /home/data/
   ```

5. **Process your files:**
   ```bash
   docker-compose exec glm-ocr python3 ocr_to_table.py
   ```

6. **Check the results:**
   - Results will be saved in `/home/gaspar/econai/output/` on the host
   - Tables are exported as Excel (.xlsx) files
   - JSON summaries are also generated

**If you make changes to the Python script, rebuild the container:**
```bash
docker-compose down
docker-compose build
docker-compose up -d
```

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

**On the Linux host:**
```
/home/gaspar/econai/
├── data/                  # Put your JPG/PDF files here
└── output/                # Results appear here
```

**Project files:**
```
.
├── Dockerfile              # Docker image definition
├── docker-compose.yml      # Docker Compose configuration
├── setup.sh               # Setup script
├── ocr_to_table.py        # Main Python OCR script
└── README.md              # This file
```

**Inside the container:**
- `/home/data/` → mounted from `/home/gaspar/econai/data/`
- `/home/output/` → mounted from `/home/gaspar/econai/output/`

## Usage

### Processing Files

1. Place your JPG or PDF files in `/home/gaspar/econai/data/` on the host
2. Verify files are visible in container:
   ```bash
   docker-compose exec glm-ocr ls -la /home/data/
   ```
3. Run the processing script:
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

# Initialize with custom paths
extractor = GLMOCRTableExtractor(
    model_name="glm4v:9b",
    data_dir="/home/data",
    output_dir="/home/output"
)

# Process a single file
extractor.process_file("/home/data/document.pdf", output_format="excel")

# Or extract from an image
table_data = extractor.extract_table_with_glm("/home/data/table.jpg")
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
