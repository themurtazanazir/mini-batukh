import pytorch_lightning as pl 
from src.models import FinalModel
from src.dataloader import MyDataset
from src.config import data_config, augmentation_config
from src.utils import GreedyCTCDecoder

train_ds = MyDataset(data_config, augmentation_config)
train_dl = train_ds(batch_size=1, shuffle=True)


model = FinalModel(vocab_size=len(data_config["letters"])+1)

# model = model.load_from_checkpoint("../lightning_logs/version_14/checkpoints/epoch=8-step=1907.ckpt")

trainer = pl.Trainer()

o = trainer.predict(model, train_dl, ckpt_path="/home/murtaza/personal/new_batukh_data/lightning_logs/version_67/checkpoints/epoch=47-step=999.ckpt")

ctc_decoder = GreedyCTCDecoder(train_ds.datagen.idx2char)

for out in o:
    text = ctc_decoder(out[0])
    print(text)