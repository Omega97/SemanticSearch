from semanticsearch.src.database import Database


def test_database(length=200):
    db = Database("./data/raw")  # Paths to text folder and loc_db file
    print("Loaded documents:", db.list_documents())

    for sample_file_name in db.list_documents():
        print('\n')
        text = db.get_document(sample_file_name)
        print(text[:200], end='...\n' if len(text) > length else '\n')


if __name__ == "__main__":
    test_database()
