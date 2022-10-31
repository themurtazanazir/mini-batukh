import pytorch_lightning as pl 
from src.models import FinalModel
from src.dataloader import MyDataset
from src.config import data_config, augmentation_config
from pytorch_lightning.callbacks import LearningRateMonitor


lr_monitor = LearningRateMonitor(logging_interval='step')

train_ds = MyDataset(data_config, augmentation_config)
train_dl = train_ds(batch_size=2, shuffle=False)


model = FinalModel(len(data_config["letters"])+1)
model = model.load_from_checkpoint("/media/murtaza/E/mini-batukh/lightning_logs/version_20/checkpoints/epoch=19-step=100000.ckpt", vocab_size=len(data_config["letters"])+1)
# model = model.load_from_checkpoint("/home/murtaza/personal/mini_batukh/lightning_logs/version_187/checkpoints/epoch=21-step=549.ckpt", vocab_size=len(data_config["letters"])+1)

trainer = pl.Trainer(max_steps=700000, callbacks=[lr_monitor], log_every_n_steps=1)
trainer.fit(model, train_dl)

