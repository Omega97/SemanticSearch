from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import os

class Caption:

    def __init__(
        self, 
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base"),
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    ):
        self.processor = processor
        self.model = model
    
    def generate_caption(self, image_path: str) -> str:
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt")
        outputs = self.model.generate(
                **inputs
                # max_length=1000,
                # min_length=20
            )
        caption = self.processor.decode(outputs[0], skip_special_tokens=True)
        return caption
    
    def generate_captions(self, dir_path: str = ".\\data\\images\\flickr8k\\Images") -> dict[str, str]:
        images_names = os.listdir(dir_path)
        images_names.sort()
        image_desc = {}
        for image_name in images_names:
            caption = self.generate_caption(os.path.join(dir_path, image_name))
            image_desc[image_name] = caption
        return image_desc