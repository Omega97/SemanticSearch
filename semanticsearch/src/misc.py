
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
