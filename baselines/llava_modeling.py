import os
import torch
import requests

from PIL import Image
from io import BytesIO
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import CLIPVisionModel, CLIPImageProcessor, StoppingCriteria
from llava.conversation import conv_templates, SeparatorStyle
from llava.utils import disable_torch_init
from llava.model import *
from llava.model.utils import KeywordsStoppingCriteria

from baselines.base import VisITBaseModel

DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"

def load_image(image_file):
    if image_file.startswith('http') or image_file.startswith('https'):
        response = requests.get(image_file)
        image = Image.open(BytesIO(response.content)).convert('RGB')
    else:
        image = Image.open(image_file).convert('RGB')
    return image

class VisITLLaVA(VisITBaseModel):
    """LLaVA model evaluation.
    """

    def __init__(self, model_args):

        assert(
            "model_path" in model_args
            and "device" in model_args
        )
        self.model_path = model_args['model_path']
        self.device = model_args['device'] if model_args["device"] >= 0 else "cpu"

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.model = LlavaLlamaForCausalLM.from_pretrained(self.model_path, low_cpu_mem_usage=True, torch_dtype=torch.float16, use_cache=True).to(self.device)
        self.image_processor = CLIPImageProcessor.from_pretrained(self.model.config.mm_vision_tower, torch_dtype=torch.float16)
        
        self.mm_use_im_start_end = getattr(self.model.config, "mm_use_im_start_end", False)
        self.tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        if self.mm_use_im_start_end:
            self.tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)

        vision_tower = self.model.get_model().vision_tower[0]
        if vision_tower.device.type == 'meta':
            vision_tower = CLIPVisionModel.from_pretrained(vision_tower.config._name_or_path, torch_dtype=torch.float16, low_cpu_mem_usage=True).to(self.device)
            self.model.get_model().vision_tower[0] = vision_tower
        else:
            vision_tower.to(device=self.device, dtype=torch.float16)
        vision_config = vision_tower.config
        vision_config.im_patch_token = self.tokenizer.convert_tokens_to_ids([DEFAULT_IMAGE_PATCH_TOKEN])[0]
        vision_config.use_im_start_end = self.mm_use_im_start_end
        if self.mm_use_im_start_end:
            vision_config.im_start_token, vision_config.im_end_token = self.tokenizer.convert_tokens_to_ids([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN])
        self.image_token_len = (vision_config.image_size // vision_config.patch_size) ** 2

        if "v1" in self.model_path.lower():
            conv_mode = "llava_v1"
        elif "mpt" in self.model_path.lower():
            conv_mode = "mpt_multimodal"
        else:
            conv_mode = "multimodal"
        self.conv_mode = conv_mode

        if "do_sample" in model_args:
            self.do_sample = model_args['do_sample']
        else:
            self.do_sample = True
        
        if "temperature" in model_args:
            self.temperature = model_args['temperature']
        else:
            self.temperature = 0.2

        if "max_new_tokens" in model_args:
            self.max_new_tokens = model_args['max_new_tokens']
        else:
            self.max_new_tokens = 1024

    def generate(self, instruction, images):

        assert len(images) == 1
        
        query = instruction
        if self.mm_use_im_start_end:
            query = query + '\n' + DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_PATCH_TOKEN * self.image_token_len + DEFAULT_IM_END_TOKEN
        else:
            query = query + '\n' + DEFAULT_IMAGE_PATCH_TOKEN * self.image_token_len
        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], query)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        inputs = self.tokenizer([prompt])

        image = load_image(images[0])
        image_tensor = self.image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
        input_ids = torch.as_tensor(inputs.input_ids).to(self.device)

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                images=image_tensor.unsqueeze(0).half().to(self.device),
                do_sample=self.do_sample,
                temperature=self.temperature,
                max_new_tokens=self.max_new_tokens,
                stopping_criteria=[stopping_criteria])

        input_token_len = input_ids.shape[1]
        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
        outputs = self.tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()
        return outputs