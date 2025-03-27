
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
