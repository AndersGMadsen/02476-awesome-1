from tests import _PATH_DATA
import torchvision
import os
from os import listdir
import pytest

from src.data.data_utils import *

def test_datautils(slice_size = (512, 512)):

    toy_image = torch.ones(1,3,512,512)

    idxs_height, idxs_width = get_slice_idxs(toy_image.shape, slice_size)
    image_slices = slice_image(toy_image, slice_size + (3,), idxs_height, idxs_width)

    for i in range(image_slices.shape[0]):
        for j in range(image_slices.shape[1]):
            assert image_slices[i,j].shape == slice_size + (3,), "Image slice did not have correct dimensions"

    unsliced_image = unslice_images(image_slices, toy_image.shape, idxs_height, idxs_width)
    assert unsliced_image.shape == toy_image.shape, "Unsliced image dimensions did not match original image"