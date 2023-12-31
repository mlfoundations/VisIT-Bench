import torch

class VisITBaseModel(torch.nn.Module):
    def __init__(self,):
        super().__init__()

    def forward(self, instruction, images):
        return self.generate(instruction, images)
    
    def generate(self, instruction, images):
        """
        instruction: (str) a string of instruction
        images: (list) a list of image urls
        Return: (str) a string of generated response
        """
        raise NotImplementedError