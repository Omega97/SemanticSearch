from semanticsearch.scripts.loader import *

print('hey')

def test_1(path):
    """Test for content"""
    content = extract_text_from_file(path)
    print(content[:100])


def test_2():
    result = process_dir('.\\prove_loader')


if __name__ == "__main__":
    # test_1(".\\prove_loader\\prova.txt")
    # test_1(".\\prove_loader\\vi_for_statisticians.pdf")
    test_2()
