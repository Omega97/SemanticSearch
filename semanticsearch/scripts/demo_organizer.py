"""
In this demo we organize the files in a directory by applying
k-means on their semantic embedding.
"""
from semanticsearch.src.semantic_organizer import SemanticFileOrganizer


def main(root_directory="..\\..\\data\\raw", num_clusters=7):
    organizer = SemanticFileOrganizer(root_directory=root_directory,
                                      num_clusters=num_clusters)
    organizer.organize_files()


# Example usage:
if __name__ == "__main__":
    main()
