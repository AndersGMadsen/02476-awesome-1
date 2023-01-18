from models.model import UNET
import pytorch_lightning as pl
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torch import nn, optim
import torch
from torch.utils.data import DataLoader
from typing import Optional, List
import torchvision
from src.data.data_utils import get_slice_idxs, slice_image, unslice_images
from tqdm import tqdm


class SegmentImage:
    def __init__(self, model: LightningModule, checkpoint_path = None):
        if checkpoint_path:
            self.model = model.load_from_checkpoint(checkpoint_path)
        else:
            self.model = model()
    
        self.model.eval()
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        torch.device(self.device)
        self.model.to(self.device)

    def __call__(self, image):
        assert len(image.shape) == 3
        assert image.shape[0] == 3
        image = torch.moveaxis(image, 0, -1)
        idxs_height, idxs_width = get_slice_idxs(image.shape, (512, 512))
        image_slices = slice_image(image, (512, 512, 3), idxs_height, idxs_width)
        image_slices = torch.moveaxis(image_slices, -1, 2)
        shape = image_slices.shape

        prediction = torch.empty((*shape[:2], 6, *shape[3:]))

        pbar = tqdm(total = shape[0] * shape[1])
        with torch.no_grad():
            for i in range(shape[0]):
                for j in range(shape[1]):
                    prediction[i, j] = self.model(image_slices[None, i, j].to(self.device)).to('cpu')
                    pbar.update(1)

        prediction = torch.moveaxis(prediction, 2, -1)
        mask_prob = unslice_images(prediction, (*image.shape[:2], 6), idxs_height, idxs_width)
        mask_prob = torch.moveaxis(mask_prob, -1, 0)
        mask_pred = mask_prob.argmax(axis=0)

        return mask_prob

if __name__ == "__main__":
    predict = SegmentImage(UNET, "models/checkpoints/epoch=2-step=21664.ckpt")
    image_path = "data/raw/images/TCGA-A2-A0D0-DX1_xmin68482_ymin39071_MPP-0.2500.png"
    image = torchvision.io.read_image(image_path).to(torch.float) / 255.0
    #print(image.shape)
    #print(image.dtype)
    predict(image)