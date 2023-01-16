import pytorch_lightning as pl
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms

from timm import create_model

class UNET(LightningModule):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x):
        pass

    def configure_optimizers(self) -> Any:
        return super().configure_optimizers()

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return super().train_dataloader()

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return super().val_dataloader()

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return super().test_dataloader()

    def training_step(self, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        return super().training_step(*args, **kwargs)

    def validation_step(self, *args: Any, **kwargs: Any) -> Optional[STEP_OUTPUT]:
        return super().validation_step(*args, **kwargs)

    def training_step(self, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        return super().training_step(*args, **kwargs)