import torch
from models.model import UNET

def test_model():

    toy_image = torch.ones(1,3,512,512)

    model = UNET()
    assert model(toy_image).shape == torch.Size([1, 6, 512, 512])