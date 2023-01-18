from tests import _PATH_DATA
import torchvision
import os
from os import listdir
import pytest

from src.data.data_utils import *

@pytest.mark.skipif(len(os.listdir(_PATH_DATA + '/raw/images')) < 2, reason="Data files not found")
def test_datautils():

    raw_dir = _PATH_DATA + '/raw'
    name_dict = split_data(raw_dir)
    image = torch.moveaxis(torchvision.io.read_image(raw_dir+'/images/'+name_dict['train'][0]), 0, -1)

    slice_size = (512, 512)

    idxs_height, idxs_width = get_slice_idxs(image.shape, slice_size)
    image_slices = slice_image(image, slice_size + (3,), idxs_height, idxs_width)

    for i in range(image_slices.shape[0]):
        for j in range(image_slices.shape[1]):
            assert image_slices[i,j].shape == slice_size + (3,), "Image slice did not have correct dimensions"

    unsliced_image = unslice_images(image_slices, image.shape, idxs_height, idxs_width)
    assert unsliced_image.shape == image.shape, "Unsliced image dimensions did not match original image"