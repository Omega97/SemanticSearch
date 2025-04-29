import os
import pandas as pd
import json
import fitz  # PyMuPDF
from semanticsearch.src.captions import *

# function to import using *
__all__ = [
    'get_extension',
    'read_txt_file',
    'read_csv_file',
    'read_pdf_file',
    'read_image_file',
]

caption_generator = CaptionGenerator()

def get_extension(file_path: str) -> str:
    """Return extension of file given path"""
    return file_path.split('.')[-1]

def read_jsonl(filepath):
    """
    Reads a JSON Lines (jsonl) file and returns a list of dictionaries.

    Args:
        filepath (str): The path to the JSONL file.

    Returns:
        list: A list of dictionaries, where each dictionary represents a line in the file.
    """
    data = []
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    json_object = json.loads(line)
                    data.append(json_object)
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON on line: {line.strip()}. Error: {e}")

    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")

    return data

def read_txt_file(file_path: str) -> str:
    """
    Reads a txt file and returns its content
    """
    with open(file_path, 'r') as f:
        content = f.read()
    return content

def read_csv_file(file_path: str, num_rows: int = 10) -> str:
    """
    Reads a csv file and returns th string of the top num_rows rows
    """
    try:
        df = pd.read_csv(file_path, nrows=num_rows)
        csv_string = df.to_csv(index=False).strip()
        return csv_string
    except Exception as e:
        return f"Error reading CSV: {e}"

def read_pdf_file(file_path: str) -> str:
    """
    Reads a pdf file and returns its content
    """
    with fitz.open(file_path) as doc:
        return "".join(page.get_text() for page in doc)
    
def read_image(file_path: str, device: torch.device = torch.device('cpu')) -> str:
    """
    Reads a image file and returns a caption for the image

    Args:
        filepath (str): The path to the txt file
        device (torch.device): device used for the caption generator model
    """

    caption_generator.to(device)
    caption = caption_generator.generate_caption(file_path)
    return caption
