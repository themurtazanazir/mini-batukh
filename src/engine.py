import pytorch_lightning as pl 
from src.models import FinalModel
from src.dataloader import MyDataset
from src.config import data_config, augmentation_config


train_ds = MyDataset(data_config, augmentation_config)
train_dl = train_ds(batch_size=1, shuffle=False)


model = FinalModel(len(data_config["letters"])+1)

trainer = pl.Trainer(max_steps=1000)
trainer.fit(model, train_dl, )

