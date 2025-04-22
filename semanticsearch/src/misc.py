import numpy as np
import os
import ctypes
import csv
from colorama import Fore, Style



def cprint(text, color_code='w'):
    """
    Print text in a specified color using colorama.

    Args:
        text (str): The text to print.
        color_code (str): A single character representing the color:
            'r' -> Red
            'g' -> Green
            'b' -> Blue
            'y' -> Yellow
            'm' -> Magenta
            'c' -> Cyan
            'w' -> White
            Any other value defaults to white.
    """
    # Map color codes to colorama.Fore constants
    color_map = {
        'r': Fore.RED,
        'g': Fore.GREEN,
        'b': Fore.BLUE,
        'y': Fore.YELLOW,
        'm': Fore.MAGENTA,
        'c': Fore.CYAN,
        'w': Fore.WHITE,
    }

    # Get the corresponding color or default to white
    code = color_code.lower()[0]
    color = color_map.get(code, Fore.WHITE)

    # Print the text in the specified color and reset afterward
    print(f"{color}{text}{Style.RESET_ALL}")


def pprint(text, width):
    """print the words one by one. When the row length exceeds
    the max width, go to new line."""
    words = text.split(" ")
    current_line = ""
    out = ""
    for word in words:
        if len(current_line) == 0:
            current_line = word
        elif len(current_line) + len(word) + 1 <= width:
            current_line += " " + word
        else:
            out += current_line + '\n'
            current_line = word
        if '\n' in word:
            out += current_line + ' '
            current_line = ""
    if current_line:
        out += current_line + '\n'
    return out


def find_element_index(vector, element):
    """
    Finds the index of an element in a list (vector).

    Args:
        vector: The list to search.
        element: The element to find.

    Returns:
        The index of the element if found, or -1 if not found.
    """
    try:
        return vector.index(element)
    except ValueError:
        return -1


def get_extension(path: str) -> str:
    """
    Gets the file extension using os.path.splitext and returns it in lowercase.
    """
    root, ext = os.path.splitext(path)
    return ext[1:].lower()  # Remove the leading dot and convert to lowercase


def get_file_name(path: str) -> str:
    """
    Gets the file name (without extension) using os.path.basename and os.path.splitext.
    """
    base_name = os.path.basename(path)  # Get the file name with extension
    name, ext = os.path.splitext(base_name)  # Split into name and extension
    return name


def set_file_hidden(file_path: str):
    """
    Set the hidden attribute using the `attrib` command
    """
    os.system(f'attrib +h "{file_path}"')


def is_file_hidden(file_path: str) -> bool:
    """
    Checks if a file is hidden on Windows by examining its attributes.

    Args:
        file_path (str): The path to the file.

    Returns:
        bool: True if the file is hidden, False otherwise.
    """
    # Get the full absolute path of the file
    full_path = os.path.abspath(file_path)

    # Call the Windows API function GetFileAttributesW
    attrs = ctypes.windll.kernel32.GetFileAttributesW(full_path)

    # Check if the function failed (invalid file path, etc.)
    if attrs == -1:
        raise FileNotFoundError(f"File not found: {file_path}")

    # Check if the FILE_ATTRIBUTE_HIDDEN flag is set
    return bool(attrs & 0x02)  # 0x02 = FILE_ATTRIBUTE_HIDDEN


def read_txt_file(file_path) -> str:
    """Read txt file"""
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()


def read_csv_file(file_path: str, n: int = 2) -> str:
    """
    Reads the first 'n' lines of a CSV file using the csv module and returns them as text.

    Args:
        file_path (str): The path to the CSV file.
        n (int): The number of lines to read.

    Returns:
        str: The first 'n' lines of the CSV file as text.
    """
    if n <= 0:
        raise ValueError("The number of lines 'n' must be greater than 0.")

    result = []
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            reader = csv.reader(file)
            for i, row in enumerate(reader):
                if i >= n:
                    break
                result.append(','.join(row) + '\n')  # Join row elements with commas
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")

    return '\n'.join(result)


def cosine_similarity(a, b):
    return np.dot(a, b.T)
