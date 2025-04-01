print('Importing...')
import numpy as np
from semanticsearch.src.embedding import EmbeddingModel


def display_colored_strings(strings, numbers,
                            foreground_rgb=(0, 200, 0),
                            background_rgb=(40, 40, 20)):
    """
    Displays strings with background colors based on associated numbers.

    Args:
        strings: A list of strings.
        numbers: A list of numbers (floats) between 0 and 1, with the same length as strings.
        foreground_rgb: color of the foreground
        background_rgb: color of the background
    """

    if len(strings) != len(numbers):
        raise ValueError("Strings and numbers lists must have the same length.")

    rbg_0 = np.array(foreground_rgb)
    rbg_1 = np.array(background_rgb)

    for i in range(len(strings)):
        string = strings[i]
        number = numbers[i]

        # Calculate RGB values for the color
        rgb = rbg_1 + number * (rbg_0 - rbg_1)
        red = int(rgb[0])
        green = int(rgb[1])
        blue = int(rgb[2])

        # Display the string with the calculated background color.
        print(f"\033[48;2;{red};{green};{blue}m{string}\033[0m")


def test_model(query, path, subtract=None):
    print('Loading model...')
    model = EmbeddingModel()

    print('Computing...')
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()

    strings = text.split('\n')
    strings = [s for s in strings if s]

    mat = model.encode(strings)

    query_emb = model.encode([query])[0]

    if subtract is not None:
        minus_emb = np.average(model.encode(subtract), axis=0)[0]
        query_emb -= minus_emb

    similarity = mat @ query_emb
    similarity -= np.average(similarity)
    similarity /= np.std(similarity)
    similarity = np.clip(similarity, 0, 1)

    print(f'\nSearching for: {query}\n')
    display_colored_strings(strings, similarity)


if __name__ == "__main__":
    test_model(query="The full story behind the invention",
               path='..\\data\\raw\\Wikipedia\\ChatGPT.txt',
               subtract = ["Philosophy", "Glossary", "Applications", "Approaches"])
