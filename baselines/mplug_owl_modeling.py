"""
Refer to https://github.com/X-PLUG/mPLUG-Owl to prepare environments and checkpoints
"""

import torch
from PIL import Image
import requests
from io import BytesIO

from base import VisITBaseModel

from mplug_owl_utils.modeling_mplug_owl import MplugOwlForConditionalGeneration
from mplug_owl_utils.processing_mplug_owl import MplugOwlImageProcessor, MplugOwlProcessor
from mplug_owl_utils.tokenization_mplug_owl import MplugOwlTokenizer


device = torch.device("cuda") if torch.cuda.is_available() else "cpu"


class VisITmPLUGOwl(VisITBaseModel):
    def __init__(self,):
        super().__init__()

        self.pretrained_ckpt = 'MAGAer13/mplug-owl-llama-7b'
        self.model = MplugOwlForConditionalGeneration.from_pretrained(
            self.pretrained_ckpt,
            torch_dtype=torch.bfloat16,
        ).to(device)
        self.image_processor = MplugOwlImageProcessor.from_pretrained(self.pretrained_ckpt)
        self.tokenizer = MplugOwlTokenizer.from_pretrained(self.pretrained_ckpt)
        self.processor = MplugOwlProcessor(self.image_processor, self.tokenizer)

    def generate(self, instruction, images):
        """
        instruction: (str) a string of instruction
        images: (list) a list of image urls
        Return: (str) a string of generated response
        """

        if len(images) == 0:
            raise ValueError('No image is provided.')
        
        generate_kwargs = {
            'do_sample': True,
            'top_k': 5,
            'max_length': 512
        }

        # prepare instruction prompt
        sep = '\n'
        prompts = [
        f'''The following is a conversation between a curious human and AI assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.
{sep.join(['Human: <image>'] * len(images))}
Human: {instruction}
AI: ''']

        # download image image
        image_inputs = []
        for img_url in images:
            response = requests.get(img_url)
            image_inputs.append(Image.open(BytesIO(response.content)).convert("RGB"))
        
        inputs = self.processor(text=prompts, images=image_inputs, return_tensors='pt')
        inputs = {k: v.bfloat16() if v.dtype == torch.float else v for k, v in inputs.items()}
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        with torch.no_grad():
            res = self.model.generate(**inputs, **generate_kwargs)
        out = self.tokenizer.decode(res.tolist()[0], skip_special_tokens=True)

        return out
    

if __name__ == '__main__':
    # test
    from test_cases import TEST_CASES

    model = VisITmPLUGOwl()

    for test_case in TEST_CASES:
        pred = model.generate(
            instruction=test_case['instruction'],
            images=test_case['images'],
        )
        print(f'Instruction:\t{test_case["instruction"]}')
        print(f'Images:\t{test_case["images"]}')
        print(f'Answer:\t{pred}')
        print('-'*20)
