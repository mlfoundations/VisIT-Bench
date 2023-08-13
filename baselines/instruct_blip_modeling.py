"""
Refer to https://huggingface.co/Salesforce/instructblip-vicuna-13b
"""

from PIL import Image
import requests
from io import BytesIO
import torch
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
# from lavis.models import load_model_and_preprocess

from .base import VisITBaseModel


device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

class VisITInstructBLIP2(VisITBaseModel):
    def __init__(self):
        super().__init__()
        # self.model, self.vis_processors, _ = load_model_and_preprocess(name="blip2_vicuna_instruct", model_type="vicuna13b", is_eval=True, device=device)
        self.model = InstructBlipForConditionalGeneration.from_pretrained("Salesforce/instructblip-vicuna-13b").to(device)
        self.processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-vicuna-13b")

    def generate(self, instruction, images):
        """
        instruction: (str) a string of instruction
        images: (list) a list of image urls
        Return: (str) a string of generated response
        """
        image_url = images[0]  # instruct_blip2 only supports single image
        response = requests.get(image_url)
        raw_image = Image.open(BytesIO(response.content)).convert("RGB")
        inputs = self.processor(images=raw_image, text=instruction, return_tensors="pt").to(device)
        # image = self.vis_processors["eval"](raw_image).unsqueeze(0).to(device)
        # out = self.model.generate({"image": image, "prompt": instruction})[0]
        outputs = self.model.generate(
                **inputs,
                do_sample=False,
                num_beams=5,
                max_length=256,
                min_length=1,
                top_p=0.9,
                repetition_penalty=1.5,
                length_penalty=1.0,
                temperature=1,
        )
        outputs[outputs == 0] = 2
        out = self.processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
        return out