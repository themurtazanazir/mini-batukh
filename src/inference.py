import pytorch_lightning as pl 
from src.models import FinalModel
from src.dataloader import MyDataset
from src.config import data_config, augmentation_config
from src.utils import GreedyCTCDecoder
from PIL import Image
from torchvision.transforms import ToTensor

train_ds = MyDataset(data_config, augmentation_config)
train_dl = train_ds(batch_size=1, shuffle=True)


model = FinalModel(vocab_size=len(data_config["letters"])+1)

# model = model.load_from_checkpoint("../lightning_logs/version_14/checkpoints/epoch=8-step=1907.ckpt")

trainer = pl.Trainer()

def gen():
    im = Image.open("data/sample3.png")
    w, h = im.size
    im = im.crop((0, 20, w, int(0.9*h)))
    new_im = train_ds.datagen.resize_and_pad_to_model_size(im)
    new_im.save("output/sample2.png")
    image = ToTensor()(new_im).unsqueeze(0)
    text ="ghjk"
    yield (image-image.mean())/255, [text], [len(text)]

# o = trainer.predict(model, train_dl, ckpt_path="/media/murtaza/E/mini-batukh/lightning_logs/version_5/checkpoints/epoch=15-step=80000.ckpt")
o = trainer.predict(model, gen(), ckpt_path="/media/murtaza/E/mini-batukh/lightning_logs/version_22/checkpoints/epoch=50-step=255000.ckpt")
# print(len(o))
ctc_decoder = GreedyCTCDecoder(train_ds.datagen.idx2char)

for out in o:
    # print(out.shape)
    text = ctc_decoder(out[:, 0, :])
    print(text)