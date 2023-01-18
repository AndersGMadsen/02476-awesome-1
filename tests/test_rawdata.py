from tests import _PATH_DATA
import torchvision
import os
from os import listdir
import pytest

from src.data.data_utils import split_data


@pytest.mark.skipif(len(os.listdir(_PATH_DATA + '/raw/images')) < 2, reason="Data files not found")
def test_rawdata():

    raw_dir = _PATH_DATA + '/raw'
    name_dict = split_data(raw_dir)
    
    # test number of images
    assert len(name_dict['train']) == 120, "Train data did not have correct number of images"
    assert len(name_dict['validation']) == 15, "Train data did not have correct number of images"
    assert len(name_dict['test']) == 16, "Test data did not have correct number of images"

    # test dimensionen of one image (takes too long to test all - maybe add this)
    first_image = torchvision.io.read_image(raw_dir+'/images/'+name_dict['train'][0])
    assert first_image.shape[0] == 3, "Fist image was not 3-dimensional"

    # test that every image has a corresponding mask
    image_names = sorted([image for image in listdir(raw_dir+'/images') if image.endswith('.png')])
    mask_names  = sorted([image for image in listdir(raw_dir+'/masks') if image.endswith('.png')])
    assert image_names == mask_names, "Image names did not match mask names"