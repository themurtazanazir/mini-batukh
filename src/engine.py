import pytorch_lightning as pl 
from src.models import FinalModel
from src.dataloader import MyDataset
from src.config import data_config, augmentation_config


train_ds = MyDataset(data_config, augmentation_config)
train_dl = train_ds(batch_size=1, shuffle=True)


model = FinalModel()

trainer = pl.Trainer()
trainer.fit(model, train_dl, )

