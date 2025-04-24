import os
import csv
import fitz  # PyMuPDF
import pandas as pd
from pathlib import Path

# Set parameters
INPUT_DIR = "prove_loader"
OUTPUT_CSV = "output.csv"
CSV_PREVIEW_LINES = 5

def csv_top_k_as_string(filepath: str, k: int = 5) -> str:
    """Turn first k rows of pandas dataframe into a string"""
    try:
        df = pd.read_csv(filepath, nrows=k)
        csv_string = df.to_csv(index=False).strip()
        return csv_string
    except Exception as e:
        return f"Error reading CSV: {e}"

def get_extension(file_path: str) -> str:
    """Returns extension of a file"""
    index = len(file_path)-file_path[::-1].find('.')
    return file_path[index:]

def extract_text_from_file(file_path: str, csv_rows: int = 10):
    ext = get_extension(file_path)

    if ext == 'txt':
        with open(file_path, 'r') as f:
            content = f.read()
        return content

    elif ext == 'pdf':
        content = ""
        with fitz.open(file_path) as doc:
            for page in doc:
                content += page.get_text()
        return content
    
    elif ext == 'csv':
        content = csv_top_k_as_string(file_path, k=csv_rows)
        return content

    elif ext in {'.jpg', '.png'}:
        try:
            img = Image.open(file_path)
            return pytesseract.image_to_string(img)
        except Exception as e:
            return f"Error processing image: {e}"
    else:
        return None

def process_dir(dir_path, csv_rows: int = 10):
    output_file = "output.csv"

    # Write header once (overwrite or create)
    with open(output_file, "w", newline='', encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "filename", "content"], quoting=csv.QUOTE_ALL)
        writer.writeheader()

    # Append rows
    for file_name in os.listdir(dir_path):
        file_path = os.path.join(dir_path, file_name)
        content = extract_text_from_file(file_path, csv_rows)
        if content is not None:
            with open(output_file, "a", newline='', encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["path", "filename", "content"], quoting=csv.QUOTE_ALL)
                writer.writerow({
                    "path": file_path,
                    "filename": file_name,
                    "content": content.replace("\n", "\\n")  # Escape newlines
                })



# def process_directory(directory):
#     rows = []
#     for root, _, files in os.walk(directory):
#         for file in files:
#             path = Path(root) / file
#             if path.suffix.lower() in {'.txt', '.pdf', '.csv', '.jpg', '.png'}:
#                 content = extract_text_from_file(path)
#                 rows.append([str(path), file, content])
#     return rows

# Run processing and write to CSV
# rows = process_directory(INPUT_DIR)
# with open(OUTPUT_CSV, 'w', newline='', encoding='utf-8') as f:
#     writer = csv.writer(f)
#     writer.writerow(['path', 'file_name', 'content'])
#     writer.writerows(rows)
