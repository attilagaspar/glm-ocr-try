#!/usr/bin/env python3
"""
Process JSON files containing firm data and convert to RAG-friendly text format
Uses OpenAI API to restructure data for better semantic search
"""

import os
import sys
import json
import re
from pathlib import Path
from typing import List, Dict, Any
import openai
from openai import OpenAI

def natural_sort_key(filename: str) -> tuple:
    """
    Sort filenames naturally (page_1, page_2, ..., page_10, page_11)
    instead of alphabetically (page_1, page_10, page_11, page_2)
    """
    parts = re.split(r'(\d+)', filename)
    return tuple(int(part) if part.isdigit() else part for part in parts)

def find_page_json_files(input_folder: Path) -> List[Path]:
    """
    Recursively find all JSON files matching pattern 'page_N.json'
    Returns sorted list in natural order
    """
    page_files = []
    
    for json_file in input_folder.rglob("*.json"):
        # Check if filename matches page_N pattern
        if re.match(r'page_\d+\.json$', json_file.name):
            page_files.append(json_file)
    
    # Sort naturally
    page_files.sort(key=lambda x: natural_sort_key(x.name))
    return page_files

def extract_firm_records(json_data: Dict) -> List[str]:
    """
    Extract firm records from JSON data
    Looks for 'shapes' list and extracts 'openai_outputs' from each element
    """
    firm_records = []
    
    if 'shapes' not in json_data:
        return firm_records
    
    for shape in json_data['shapes']:
        if isinstance(shape, dict) and 'openai_outputs' in shape:
            openai_output = shape['openai_outputs']
            if openai_output:  # Not empty
                firm_records.append(openai_output)
    
    return firm_records

def format_for_rag(firm_data: str, model: str, client: OpenAI) -> str:
    """
    Send firm data to OpenAI API to reformat for RAG
    """
    prompt = """You are a data formatting assistant for a historical economic research RAG (Retrieval-Augmented Generation) system.

Your task: Convert the raw firm data below into well-structured, readable text optimized for semantic search and retrieval.

REQUIREMENTS:
1. Use clear section headers (COMPANY INFO, LEADERSHIP, FINANCIALS, etc.)
2. Write in complete sentences where appropriate
3. List names clearly without excessive parenthetical notes
4. Highlight key facts: founding year, location, industry, leaders
5. Remove parsing artifacts and clean up formatting
6. Make it easy for someone to quickly scan and find information
7. Keep all factual information - do not invent or omit data
8. Use consistent date formats
9. Group related information together

OUTPUT ONLY THE FORMATTED TEXT. DO NOT include explanations, metadata, or commentary.

RAW FIRM DATA:
{firm_data}

FORMATTED TEXT:"""

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a precise data formatting assistant. Return only the formatted text, nothing else."},
                {"role": "user", "content": prompt.format(firm_data=firm_data)}
            ],
            temperature=0.3,
            max_tokens=2000
        )
        
        formatted_text = response.choices[0].message.content.strip()
        return formatted_text
        
    except Exception as e:
        print(f"  ERROR calling OpenAI API: {e}")
        return f"[ERROR FORMATTING]\n{firm_data}"

def process_json_files(input_folder: str, model: str, api_key: str, output_folder: str):
    """
    Main processing function
    """
    input_path = Path(input_folder)
    output_path = Path(output_folder)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Initialize OpenAI client
    client = OpenAI(api_key=api_key)
    
    # Find all page JSON files
    print(f"Scanning for page_N.json files in: {input_path}")
    json_files = find_page_json_files(input_path)
    
    if not json_files:
        print("No page_N.json files found!")
        return
    
    print(f"Found {len(json_files)} page files\n")
    
    # Process each file
    total_firms = 0
    
    for file_idx, json_file in enumerate(json_files, 1):
        print(f"[{file_idx}/{len(json_files)}] Processing: {json_file.name}")
        
        # Read JSON
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
        except Exception as e:
            print(f"  ERROR reading JSON: {e}")
            continue
        
        # Extract firm records
        firm_records = extract_firm_records(json_data)
        
        if not firm_records:
            print(f"  No firm records found")
            continue
        
        print(f"  Found {len(firm_records)} firm record(s)")
        
        # Create output file for this page
        output_file = output_path / f"{json_file.stem}_firms.txt"
        
        # Open output file in write mode
        with open(output_file, 'w', encoding='utf-8') as out_f:
            # Process each firm record
            for firm_idx, firm_data in enumerate(firm_records, 1):
                print(f"    [{firm_idx}/{len(firm_records)}] Formatting firm record...", end='', flush=True)
                
                # Format with OpenAI
                formatted_text = format_for_rag(firm_data, model, client)
                
                # Write to file immediately
                out_f.write("="*80 + "\n")
                out_f.write(f"FIRM RECORD {total_firms + firm_idx}\n")
                out_f.write(f"Source: {json_file.name}, Record {firm_idx}\n")
                out_f.write("="*80 + "\n\n")
                out_f.write(formatted_text)
                out_f.write("\n\n")
                
                # Flush to disk
                out_f.flush()
                
                print(" ✓")
                total_firms += 1
        
        print(f"  Saved to: {output_file}\n")
    
    print("="*80)
    print(f"Processing complete!")
    print(f"Total firms processed: {total_firms}")
    print(f"Output saved to: {output_path}")
    print("="*80)

def main():
    """
    Command line interface
    """
    if len(sys.argv) < 3:
        print("Usage: python json_to_rag_text.py <input_folder> <openai_model> [output_folder] [api_key]")
        print()
        print("Arguments:")
        print("  input_folder  : Folder containing page_N.json files")
        print("  openai_model  : OpenAI model to use (e.g., gpt-4o, gpt-4-turbo, gpt-3.5-turbo)")
        print("  output_folder : (Optional) Where to save output files (default: ./rag_output)")
        print("  api_key       : (Optional) OpenAI API key (default: from OPENAI_API_KEY env var)")
        print()
        print("Example:")
        print("  python json_to_rag_text.py ./data gpt-4o ./output")
        sys.exit(1)
    
    input_folder = sys.argv[1]
    model = sys.argv[2]
    output_folder = sys.argv[3] if len(sys.argv) > 3 else "./rag_output"
    api_key = sys.argv[4] if len(sys.argv) > 4 else os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        print("ERROR: OpenAI API key not provided!")
        print("Either:")
        print("  1. Set OPENAI_API_KEY environment variable")
        print("  2. Pass as 4th argument: python json_to_rag_text.py <input> <model> <output> <api_key>")
        sys.exit(1)
    
    if not Path(input_folder).exists():
        print(f"ERROR: Input folder does not exist: {input_folder}")
        sys.exit(1)
    
    print(f"Input folder: {input_folder}")
    print(f"OpenAI model: {model}")
    print(f"Output folder: {output_folder}")
    print(f"API key: {'*' * (len(api_key) - 4) + api_key[-4:]}")
    print()
    
    process_json_files(input_folder, model, api_key, output_folder)

if __name__ == "__main__":
    main()
