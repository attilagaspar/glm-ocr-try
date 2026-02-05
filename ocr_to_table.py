#!/usr/bin/env python3
"""
GLM-OCR Table Extraction Script
Converts JPG/PDF files to structured tables using GLM-4V model via Ollama
"""

import os
import json
import base64
import subprocess
from pathlib import Path
from typing import List, Dict, Union
import pandas as pd
from PIL import Image
from pdf2image import convert_from_path


class GLMOCRTableExtractor:
    """Extract tables from images and PDFs using GLM-4V model"""
    
    def __init__(self, model_name: str = "glm4v:9b"):
        """
        Initialize the table extractor
        
        Args:
            model_name: Name of the Ollama model to use
        """
        self.model_name = model_name
        self.output_dir = Path("/workspace/output")
        self.data_dir = Path("/workspace/data")
        self.output_dir.mkdir(exist_ok=True)
        
    def ensure_ollama_running(self):
        """Ensure Ollama service is running"""
        try:
            result = subprocess.run(
                ["pgrep", "-f", "ollama serve"],
                capture_output=True,
                text=True
            )
            if result.returncode != 0:
                print("Starting Ollama service...")
                subprocess.Popen(
                    ["ollama", "serve"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                import time
                time.sleep(3)
        except Exception as e:
            print(f"Warning: Could not check Ollama status: {e}")
    
    def image_to_base64(self, image_path: Union[str, Path]) -> str:
        """
        Convert image to base64 string
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Base64 encoded image string
        """
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    def pdf_to_images(self, pdf_path: Union[str, Path]) -> List[Path]:
        """
        Convert PDF to images
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of paths to converted images
        """
        pdf_path = Path(pdf_path)
        output_folder = self.output_dir / pdf_path.stem
        output_folder.mkdir(exist_ok=True)
        
        print(f"Converting PDF to images: {pdf_path.name}")
        images = convert_from_path(pdf_path, dpi=300)
        
        image_paths = []
        for i, image in enumerate(images):
            image_path = output_folder / f"page_{i+1}.jpg"
            image.save(image_path, "JPEG")
            image_paths.append(image_path)
            
        print(f"Converted {len(images)} pages")
        return image_paths
    
    def extract_table_with_glm(self, image_path: Union[str, Path]) -> Dict:
        """
        Extract table from image using GLM-4V via Ollama
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary containing extracted table data
        """
        image_path = Path(image_path)
        print(f"\nProcessing: {image_path.name}")
        
        # Prepare the prompt for table extraction
        prompt = """Analyze this image and extract any tables you find. 
For each table:
1. Identify all columns and their headers
2. Extract all rows of data
3. Preserve the table structure

Return the result as a JSON object with this structure:
{
    "tables": [
        {
            "table_number": 1,
            "headers": ["Column1", "Column2", ...],
            "rows": [
                ["value1", "value2", ...],
                ["value1", "value2", ...]
            ]
        }
    ]
}

If no tables are found, return: {"tables": [], "note": "No tables detected"}
"""
        
        # Read and encode image
        with open(image_path, "rb") as img_file:
            image_b64 = base64.b64encode(img_file.read()).decode('utf-8')
        
        # Call Ollama API
        try:
            cmd = [
                "ollama", "run", self.model_name,
                prompt
            ]
            
            # Note: Ollama CLI doesn't directly support image input via command line
            # We need to use the API endpoint instead
            import requests
            
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "images": [image_b64],
                    "stream": False
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                response_text = result.get("response", "")
                print(f"Model response received ({len(response_text)} chars)")
                
                # Try to parse JSON from response
                try:
                    # Extract JSON from markdown code blocks if present
                    if "```json" in response_text:
                        json_start = response_text.find("```json") + 7
                        json_end = response_text.find("```", json_start)
                        response_text = response_text[json_start:json_end].strip()
                    elif "```" in response_text:
                        json_start = response_text.find("```") + 3
                        json_end = response_text.find("```", json_start)
                        response_text = response_text[json_start:json_end].strip()
                    
                    table_data = json.loads(response_text)
                    return table_data
                except json.JSONDecodeError:
                    print("Warning: Could not parse JSON response, returning raw text")
                    return {
                        "tables": [],
                        "raw_response": response_text,
                        "note": "Failed to parse structured data"
                    }
            else:
                print(f"Error: API returned status {response.status_code}")
                return {"tables": [], "error": f"API error: {response.status_code}"}
                
        except requests.exceptions.ConnectionError:
            print("Error: Could not connect to Ollama. Make sure Ollama is running.")
            return {"tables": [], "error": "Connection failed"}
        except Exception as e:
            print(f"Error calling Ollama: {e}")
            return {"tables": [], "error": str(e)}
    
    def save_tables_to_excel(self, table_data: Dict, output_path: Union[str, Path]):
        """
        Save extracted tables to Excel file
        
        Args:
            table_data: Dictionary containing table data
            output_path: Path to save the Excel file
        """
        output_path = Path(output_path)
        
        if not table_data.get("tables"):
            print("No tables to save")
            # Save raw response if available
            if "raw_response" in table_data:
                txt_path = output_path.with_suffix('.txt')
                with open(txt_path, 'w', encoding='utf-8') as f:
                    f.write(table_data["raw_response"])
                print(f"Saved raw response to: {txt_path}")
            return
        
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            for table in table_data["tables"]:
                table_num = table.get("table_number", 1)
                headers = table.get("headers", [])
                rows = table.get("rows", [])
                
                if rows:
                    df = pd.DataFrame(rows, columns=headers if headers else None)
                    sheet_name = f"Table_{table_num}"
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
                    print(f"Saved table {table_num} with {len(rows)} rows")
        
        print(f"Excel file saved: {output_path}")
    
    def save_tables_to_csv(self, table_data: Dict, output_base_path: Union[str, Path]):
        """
        Save extracted tables to CSV files
        
        Args:
            table_data: Dictionary containing table data
            output_base_path: Base path for CSV files (will add table numbers)
        """
        output_base_path = Path(output_base_path)
        
        if not table_data.get("tables"):
            print("No tables to save")
            return
        
        for table in table_data["tables"]:
            table_num = table.get("table_number", 1)
            headers = table.get("headers", [])
            rows = table.get("rows", [])
            
            if rows:
                df = pd.DataFrame(rows, columns=headers if headers else None)
                csv_path = output_base_path.parent / f"{output_base_path.stem}_table{table_num}.csv"
                df.to_csv(csv_path, index=False, encoding='utf-8')
                print(f"Saved table {table_num} to: {csv_path}")
    
    def process_file(self, file_path: Union[str, Path], output_format: str = "excel"):
        """
        Process a single file (JPG or PDF)
        
        Args:
            file_path: Path to the input file
            output_format: Output format ('excel' or 'csv')
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            print(f"Error: File not found: {file_path}")
            return
        
        print(f"\n{'='*60}")
        print(f"Processing: {file_path.name}")
        print(f"{'='*60}")
        
        # Handle PDF files
        if file_path.suffix.lower() == '.pdf':
            image_paths = self.pdf_to_images(file_path)
        else:
            image_paths = [file_path]
        
        # Process each image
        all_results = []
        for i, image_path in enumerate(image_paths):
            print(f"\nPage {i+1}/{len(image_paths)}")
            table_data = self.extract_table_with_glm(image_path)
            all_results.append({
                "page": i+1,
                "image_path": str(image_path),
                "data": table_data
            })
            
            # Save individual page results
            base_name = f"{file_path.stem}_page{i+1}"
            if output_format == "excel":
                output_path = self.output_dir / f"{base_name}.xlsx"
                self.save_tables_to_excel(table_data, output_path)
            else:
                output_path = self.output_dir / base_name
                self.save_tables_to_csv(table_data, output_path)
        
        # Save combined JSON results
        json_path = self.output_dir / f"{file_path.stem}_results.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, indent=2, ensure_ascii=False)
        print(f"\nSaved combined results to: {json_path}")
        
        print(f"\n{'='*60}")
        print(f"Processing complete for: {file_path.name}")
        print(f"{'='*60}\n")


def main():
    """Main execution function"""
    print("GLM-OCR Table Extraction Tool")
    print("="*60)
    
    # Initialize extractor
    extractor = GLMOCRTableExtractor()
    
    # Ensure Ollama is running
    extractor.ensure_ollama_running()
    
    # Find files to process
    data_dir = Path("/workspace/data")
    if not data_dir.exists():
        print(f"Error: Data directory not found: {data_dir}")
        return
    
    # Look for JPG and PDF files
    files_to_process = list(data_dir.glob("*.jpg")) + \
                      list(data_dir.glob("*.jpeg")) + \
                      list(data_dir.glob("*.pdf"))
    
    if not files_to_process:
        print(f"No JPG or PDF files found in {data_dir}")
        return
    
    print(f"\nFound {len(files_to_process)} file(s) to process:")
    for f in files_to_process:
        print(f"  - {f.name}")
    
    # Process each file
    for file_path in files_to_process:
        try:
            extractor.process_file(file_path, output_format="excel")
        except Exception as e:
            print(f"Error processing {file_path.name}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*60)
    print("All files processed!")
    print(f"Results saved to: {extractor.output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
