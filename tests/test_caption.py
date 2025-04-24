import os
from semanticsearch.src.captions import *

print("Loading CaptionGenerator ...")
capgen = CaptionGenerator()
print('CaptionGenerator loaded\n')

def test_1(image_path: str = r".\data\images\flickr8k\dev10\17273391_55cfc7d3d4.jpg"):
    """
    Test caption generation on single image
    """

    print("TEST 1")
    caption = capgen.generate_caption(image_path)
    
    print(f'caption : {caption}\n\n')

def test_2(dir_path: str = r".\data\images\flickr8k\dev10"):
    """
    Test caption generator on directory of images
    """
    print("TEST 2")
    captions_dict = capgen.generate_captions(dir_path)
    i = 0
    print('captions:')
    for name in captions_dict.keys():
        caption = captions_dict[name]
        print(f'{i}. {name} -> {caption}')
        i += 1


if __name__ == '__main__':
    test_1()
    test_2()
