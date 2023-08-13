"""
Refer to https://github.com/microsoft/TaskMatrix to prepare environments and checkpoints
"""

import os
import torch

from visual_chatgpt import ConversationBot
from .base import VisITBaseModel


device = torch.device("cuda") if torch.cuda.is_available() else "cpu"

class VisITInstructBLIP2(VisITBaseModel):
    def __init__(self,):
        super().__init__()
        """ 
        Default ChatGPT service is text-davinci-003, the original codes fixed this model type, which is the only type that generate reasonable answers in my exp.
        you can change the model type by specifing OPENAI_MODEL_NAME in ./visual_chatgpt_utils/visual_chatgpt.py L44 -- Rulin
        """
        if not os.path.exists("checkpoints"):
            os.mkdir("checkpoints")

        load = "ImageCaptioning_cuda:0,Text2Image_cuda:0"
        load_dict = {e.split('_')[0].strip(): e.split('_')[1].strip() for e in load.split(',')}
        self.bot = ConversationBot(load_dict=load_dict)

    def generate(self, instruction, images):
        """
        instruction: (str) a string of instruction
        images: (list) a list of image urls
        Return: (str) a string of generated response
        """
        self.bot.init_agent('English')    # init a new env for each input, otherwise the previous samples will be recorded in memory
        try:
            for image in images:
                _ = self.bot.run_image(image, [], '', 'English')
            txt_outs = self.bot.run_text(instruction, [])
            out = txt_outs[0].split('The image file name is')[0]
        except:
            out = 'I can not answer your question due to unknown action.'    # at a very low chance, the bot might fail to detect correct action and raise error 
        return out
