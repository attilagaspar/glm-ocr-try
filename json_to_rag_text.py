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
    Skips records without a firm name
    """
    firm_records = []
    
    if 'shapes' not in json_data:
        return firm_records
    
    for shape in json_data['shapes']:
        if isinstance(shape, dict) and 'openai_outputs' in shape:
            openai_output = shape['openai_outputs']
            if openai_output:  # Not empty
                # Convert to string if it's a list or other type
                if isinstance(openai_output, list):
                    openai_output = ' '.join(str(item) for item in openai_output)
                elif not isinstance(openai_output, str):
                    openai_output = str(openai_output)
                
                # Skip if no firm name (check for common indicators)
                openai_lower = openai_output.lower()
                # Skip if it looks like it has no firm name
                if 'firm name' in openai_lower and ('unknown' in openai_lower or 'missing' in openai_lower or 'n/a' in openai_lower):
                    continue
                # Also check if the text is just empty/whitespace or very short
                if len(openai_output.strip()) < 10:
                    continue
                firm_records.append(openai_output)
    
    return firm_records

def format_for_rag(firm_data: str, model: str, client: OpenAI, source_name: str = None, prompt_file: str = "rag_formatting_prompt.txt") -> str:
    """
    Send firm data to OpenAI API to reformat for RAG
    """
    source_instruction = ""
    if source_name:
        source_instruction = f"\n7. Begin the output with: 'Source: {source_name}'"
    
    # Load prompt from file
    prompt_path = Path(__file__).parent / prompt_file
    try:
        with open(prompt_path, 'r', encoding='utf-8') as f:
            prompt = f.read()
    except FileNotFoundError:
        print(f"❌ ERROR: Prompt file not found: {prompt_path}")
        return f"[ERROR: PROMPT FILE NOT FOUND]\n{firm_data}"
    
    full_prompt = prompt.format(firm_data=firm_data, source_instruction=source_instruction)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a precise data formatting assistant. Return only the formatted text, nothing else."},
                {"role": "user", "content": full_prompt}
            ],
            temperature=0.3,
            max_tokens=2000
        )
        
        formatted_text = response.choices[0].message.content.strip()
        
        print("\n" + "─"*80)
        print("📥 OPENAI RESPONSE:")
        print("─"*80)
        print(formatted_text)
        print("─"*80 + "\n")
        
        return formatted_text
        
    except Exception as e:
        print(f"❌ ERROR calling OpenAI API: {e}\n")
        return f"[ERROR FORMATTING]\n{firm_data}"

def process_json_files(input_folder: str, model: str, api_key: str, output_folder: str, source_name: str = None):
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
    
    print(f"Found {len(json_files)} page files")
    if source_name:
        print(f"Source attribution: {source_name}")
    print()
    
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
        # Include parent folder name to avoid overwrites when processing recursively
        parent_folder = json_file.parent.name
        output_file = output_path / f"{parent_folder}_{json_file.stem}_firms.txt"
        
        # Open output file in write mode
        with open(output_file, 'w', encoding='utf-8') as out_f:
            # Process each firm record
            for firm_idx, firm_data in enumerate(firm_records, 1):
                print(f"    [{firm_idx}/{len(firm_records)}] Formatting firm record...", end='', flush=True)
                
                # Format with OpenAI (prompt loaded from rag_formatting_prompt.txt)
                formatted_text = format_for_rag(firm_data, model, client, source_name)
                
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
        print("Usage: python json_to_rag_text.py <input_folder> <openai_model> [output_folder] [source_name] [api_key]")
        print()
        print("Arguments:")
        print("  input_folder  : Folder containing page_N.json files")
        print("  openai_model  : OpenAI model to use (e.g., gpt-4o, gpt-4-turbo, gpt-3.5-turbo)")
        print("  output_folder : (Optional) Where to save output files (default: ./rag_output)")
        print("  source_name   : (Optional) Source attribution (e.g., '1896 Compass', 'Budapest Stock Exchange Yearbook 1920')")
        print("  api_key       : (Optional) OpenAI API key (default: from OPENAI_API_KEY env var)")
        print()
        print("Examples:")
        print("  python json_to_rag_text.py ./data gpt-4o")
        print("  python json_to_rag_text.py ./data gpt-4o ./output '1896 Compass'")
        print("  python json_to_rag_text.py ./data gpt-4o ./output '1920 Budapest Stock Exchange Yearbook' sk-...")
        sys.exit(1)
    
    input_folder = sys.argv[1]
    model = sys.argv[2]
    output_folder = sys.argv[3] if len(sys.argv) > 3 else "./rag_output"
    source_name = sys.argv[4] if len(sys.argv) > 4 else None
    api_key = sys.argv[5] if len(sys.argv) > 5 else os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        print("ERROR: OpenAI API key not provided!")
        print("Either:")
        print("  1. Set OPENAI_API_KEY environment variable")
        print("  2. Pass as 5th argument")
        sys.exit(1)
    
    if not Path(input_folder).exists():
        print(f"ERROR: Input folder does not exist: {input_folder}")
        sys.exit(1)
    
    print(f"Input folder: {input_folder}")
    print(f"OpenAI model: {model}")
    print(f"Output folder: {output_folder}")
    if source_name:
        print(f"Source name: {source_name}")
    print(f"API key: {'*' * (len(api_key) - 4) + api_key[-4:]}")
    print()
    
    process_json_files(input_folder, model, api_key, output_folder, source_name)

if __name__ == "__main__":
    main()
