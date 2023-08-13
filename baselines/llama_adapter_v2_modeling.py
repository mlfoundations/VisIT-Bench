"""
Refer to https://github.com/VegB/LLaMA-Adapter/tree/main/llama_adapter_v2_multimodal to prepare environments and checkpoints
"""

import torch
from PIL import Image
import requests
from io import BytesIO

from base import VisITBaseModel

import llama_adapter_v2_utils as llama


LLAMA_DIR = 'REPLACE_WITH_LOCAL_LLAMA_7B_DIR'
device = torch.device("cuda") if torch.cuda.is_available() else "cpu"


class VisITLlamaAdapterV2(VisITBaseModel):
    def __init__(self,):
        super().__init__()
        self.model, self.preprocess = llama.load("BIAS-7B", LLAMA_DIR, device)

    def generate(self, instruction, images):
        """
        instruction: (str) a string of instruction
        images: (list) a list of image urls
        Return: (str) a string of generated response
        """

        if len(images) == 0:
            raise ValueError('No image is provided.')
        if len(images) > 1:
            return '[Skipped]: Currently only support single image.'

        # download image image
        response = requests.get(images[0])
        img = Image.open(BytesIO(response.content)).convert("RGB")
        img = self.preprocess(img).unsqueeze(0).to(device)
        
        # predict
        out = self.model.generate(img, [instruction])[0]

        return out
    

if __name__ == '__main__':
    # test
    from test_cases import TEST_CASES

    model = VisITLlamaAdapterV2()

    for test_case in TEST_CASES:
        pred = model.generate(
            instruction=test_case['instruction'],
            images=test_case['images'],
        )
        print(f'Instruction:\t{test_case["instruction"]}')
        print(f'Images:\t{test_case["images"]}')
        print(f'Answer:\t{pred}')
        print('-'*20)
