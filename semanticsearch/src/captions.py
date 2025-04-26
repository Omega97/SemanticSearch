from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import os
import torch
from typing import Optional

class CaptionGenerator:
    """
    Class used to generate captions from single or directory of images
    """

    def __init__(
        self, 
        device: Optional[torch.device] = None,
        processor = None,
        model = None,
    ):
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device
            
        self.processor = processor if processor is not None else BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base", use_fast=True)
        self.model = model if model is not None else BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(self.device)

    def generate_caption(self, image_path: str) -> str:
        """Given the path of an image, returns its description as a string"""

        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        outputs = self.model.generate(**inputs)
        caption = self.processor.decode(outputs[0], skip_special_tokens=True)
        return caption
    
    def generate_captions(self, dir_path: str = ".\\data\\images\\flickr8k\\Images") -> dict[str, str]:
        """Given a directory returns the dictionary of pairs image_name and image caption"""
        captions = {}

        for image_name in sorted(os.listdir(dir_path)):
            image_path = os.path.join(dir_path, image_name)
            caption = self.generate_caption(image_path)
            captions[image_name] = caption
        return captions

    def to(self, device: torch.device) -> None:
        """Updates device attribute and changes device of model"""
        self.device = device
        self.model.to(device)
        