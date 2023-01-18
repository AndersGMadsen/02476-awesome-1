import hydra
from models.model import UNET
import torch
from tqdm import tqdm
from pytorch_lightning import Trainer
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint


@hydra.main(version_base=None, config_path='config', config_name='default_config.yaml')
def train(config):	
	hparams = config.experiment

	torch.cuda.empty_cache()
	torch.manual_seed(hparams['seed'])

	model = UNET(batch_size = hparams['batch_size'], lr = hparams['lr'], data_dir=hparams['data_dir'])

	early_stopping_callback = EarlyStopping(
		monitor='val_loss', min_delta = 0.0, patience=3, verbose=True, mode='min')

	checkpoint_callback = ModelCheckpoint(dirpath = 'models/checkpoints/')

	if hparams["wandb"]:
		logger = pl.loggers.WandbLogger(project='02476-awesome-1')
	else:
		logger = None

	trainer = Trainer(callbacks=[checkpoint_callback, early_stopping_callback],
					  logger=logger,
					  max_epochs=hparams['max_epochs'], gpus=hparams['gpus'])

	trainer.fit(model)

if __name__ == "__main__":
	train()