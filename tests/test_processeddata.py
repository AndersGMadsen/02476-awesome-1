from tests import _PATH_DATA
import os
import torch
import pytest


from torch.utils.data import DataLoader
from src.data.dataset import BCSSDataset



@pytest.mark.skipif(len(os.listdir(_PATH_DATA + '/processed/train')) < 0, reason="Data files not found")
def test_processeddata():

    dir = _PATH_DATA + '/processed'

    # test the correct number of images
    assert len(os.listdir(dir+'/train')) == 8832, "Train folder did not have correct number of processed images"
    assert len(os.listdir(dir+'/validation')) == 1161, "Validation folder did not have correct number of processed images"
    assert len(os.listdir(dir+'/test')) == 1124, "Test folder did not have correct number of processed images"

    train_data = BCSSDataset(root_dir=dir, key="train")
    train_loader = DataLoader(dataset=train_data, batch_size=1, shuffle=True, num_workers=6)
    image, mask = next(iter(train_loader))

    # test correct dimensions of processed data
    assert image.shape == torch.Size([1, 3, 512, 512]), "Processed images did not have correct shape"
    assert mask.shape == torch.Size([1, 6, 512, 512]), "Processed mask did not have correct shape"