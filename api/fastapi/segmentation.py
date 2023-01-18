import io

import torch
from PIL import Image
import numpy as np
from torchvision import transforms
from models.model import UNET
from src.models.predict_model import SegmentImage


def get_segmentator():
    predict = SegmentImage(UNET, "models/checkpoints/epoch=11-step=6624.ckpt")
    return predict


def get_segments(predict, binary_image, max_size=512):

    input_image = Image.open(io.BytesIO(binary_image)).convert("RGB")
    input_image = torch.from_numpy(np.array(input_image)).to(float) / 255.0
    input_image = torch.moveaxis(input_image, -1, 0)

    output = predict(input_image)

    # create a color palette, selecting a color for each class
    palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
    colors = torch.as_tensor([i for i in range(6)])[:, None] * palette
    colors = (colors % 255).numpy().astype("uint8")

    # plot the semantic segmentation predictions of 21 classes in each color
    r = Image.fromarray(output.byte().cpu().numpy())
    r.putpalette(colors)

    return r
