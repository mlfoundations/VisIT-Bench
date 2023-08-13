"""
Refer to https://github.com/Vision-CAIR/MiniGPT-4 to prepare environments and checkpoints
"""

import torch
from PIL import Image
import requests
from io import BytesIO

from base import VisITBaseModel

from minigpt4_utils.common.config import Config
from minigpt4_utils.common.dist_utils import get_rank
from minigpt4_utils.common.registry import registry
from minigpt4_utils.conversation.conversation import Chat, CONV_VISION

from minigpt4_utils.datasets.builders import *
from minigpt4_utils.models import *
from minigpt4_utils.processors import *
from minigpt4_utils.runners import *
from minigpt4_utils.tasks import *


device = torch.device("cuda") if torch.cuda.is_available() else "cpu"


class VisITMiniGPT4(VisITBaseModel):
    def __init__(self,):
        super().__init__()

        cfg = Config(cfg_path='./minigpt4_utils/minigpt4_eval.yaml')
        model_config = cfg.model_cfg
        model_cls = registry.get_model_class(model_config.arch)
        model = model_cls.from_config(model_config).to(device)

        vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
        self.vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
        self.chat = Chat(model, self.vis_processor, device=device)

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
        
        # Init chat state
        chat_state = CONV_VISION.copy()
        img_list = []

        # download image image
        response = requests.get(images[0])
        img = Image.open(BytesIO(response.content)).convert("RGB")
        
        # upload Image
        self.chat.upload_img(img, chat_state, img_list)

        # ask
        self.chat.ask(instruction, chat_state)

        # answer
        out = self.chat.answer(
            conv=chat_state,
            img_list=img_list,
            num_beams=1,
            temperature=1.0,
            max_new_tokens=300,
            max_length=2000
        )[0]

        return out


if __name__ == '__main__':
    # test
    from test_cases import TEST_CASES

    model = VisITMiniGPT4()

    for test_case in TEST_CASES:
        pred = model.generate(
            instruction=test_case['instruction'],
            images=test_case['images'],
        )
        print(f'Instruction:\t{test_case["instruction"]}')
        print(f'Images:\t{test_case["images"]}')
        print(f'Answer:\t{pred}')
        print('-'*20)
