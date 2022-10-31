from PIL import Image, ImageFont, ImageDraw
import os
import numpy as np
import random

import torch
from src.augmentations import RandomBgColor, RandomPerespective
from src.augmentations import Compose
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor
from torch.nn.utils.rnn import pad_sequence

class DataGenerator:

    def __init__(self, fonts_path, fonts, font_sizes, words_path, length,
                 aff_aug, img_aug, letters, model_in_width, model_in_height):
        self.letters = letters

        with open(words_path) as f:
            data = f.read()

        for char in data:
            if char not in self.letters:
                data = data.replace(char, '')
        
        self.words = data.strip()
        self.words = self.words.replace("\n", " ").split()
        self.fonts_path = fonts_path
        self.fonts = fonts
        self.font_sizes = font_sizes
        self.length = length

        self.idx2char = dict(enumerate(self.letters, 1))
        self.char2idx = {v: k for k, v in self.idx2char.items()}

        self.OOV_idx = max(self.idx2char.keys())+1

        self.aff_aug = aff_aug
        self.img_aug = img_aug

        self.model_in_width = model_in_width
        self.model_in_height = model_in_height


    def encode_text(self, text):
        text = [self.char2idx.get(char, self.OOV_idx) for char in text]
        return text

    def decode_text(self, text):
        return "".join([self.idx2char.get(char, '<OOV>') for char in text])

    def get_random_font(self, meta={}):

        fontFile = random.choice(self.fonts)
        meta['fontFile'] = fontFile
        fontSize = random.randint(*self.font_sizes)
        meta['fontSize'] = fontSize
        # load the font and image
        font = ImageFont.truetype(
            os.path.join(self.fonts_path, fontFile),

            fontSize,
            layout_engine=ImageFont.LAYOUT_RAQM,
        )
        return font, meta

    def generate_text(self, length, meta={}):

        # TODO: Add more types of random texts with different probability.
        text = " ".join([random.choice(self.words)
                        for _ in range(length)])
        meta["text"] = text
        return text, meta

    def crop_tight(self, image, padding_per=0.1):
        w, h = image.size
        try:
            img = np.array(image)[:, :, :3]
        except:
            print(w, h)
            raise
        vt, hz = np.where((img < 255).all(axis=-1))

        x0, y0, x1, y1 = (hz.min(), vt.min(), hz.max(), vt.max())
        padding_x = padding_per*(y1-y0)
        padding_y = padding_per*(y1-y0)

        x0 = max(0, x0-padding_x)
        x1 = min(w, x1+padding_x)
        y0 = max(0, y0-padding_y)
        y1 = min(h, y1+padding_y)

        image = image.crop((x0, y0, x1, y1))
        return image

    def draw_image(self, font, text, meta):

        w, h = font.getsize(text)#, direction='rtl')
        if w==0:
            w = 1
        if h==0:
            h=1
        # PIL gives wrong sizes at certain font sizes. just keeeping
        w = int(w+(2*w))
        # these large enough to almost always encompass the whole text
        h = int(h+(2*h))

        r_color = random.randint(0, 120)
        g_color = max(0, r_color+random.randint(-10, 10))
        b_color = max(0, r_color+random.randint(-10, 10))
        text_color = [r_color, g_color, b_color]
        random.shuffle(text_color)
        text_color = tuple(text_color)
        meta["text_color"] = text_color

        image = Image.new(mode='RGBA', size=(w, h), color='#ffffff00')
        draw = ImageDraw.Draw(image)
        draw.text((int(0.25*w), int(0.25*h)), text,
                  font=font, fill=text_color, color=text_color, direction='rtl')
        draw = ImageDraw.Draw(image)

        padding_per = random.random()*0.2+0.02
        meta["padding_per"] = padding_per

        image = self.crop_tight(image, padding_per)
        return image, meta

    def generate_clean_sample(self):

        meta = {}

        font, meta = self.get_random_font(meta)

        length = random.randint(*self.length)
        meta["length"] = length
        text = ''
        while len(text)< 5:
            text, meta = self.generate_text(length, meta)

        text = text[:20]
        # print("text is", text, len(text))
        # print(text)
        image, meta = self.draw_image(font, text, meta)

        return image, text, meta

    def resize_and_pad_to_model_size(self, image):
        w, h = image.size
        image = image.resize(
            (int(w/h*self.model_in_height), self.model_in_height))
        w, h = image.size
        image = np.array(image)

        image = image[:, ::-1]  # crop if longer
        image = image[:, :self.model_in_width]  # crop if longer
        h, w = image.shape[:2]
        image = np.pad(image, ((0, 0), (0, self.model_in_width-w), (0, 0)),
                       constant_values=0)
        return Image.fromarray(image)

    def __getitem__(self, idx):
        image, text, meta = self.generate_clean_sample()
        image, meta = RandomPerespective(max_change=25)(image, meta={})
        image = self.aff_aug(np.array(image))
        image, meta = RandomBgColor(base_color_range= (195, 255), 
        tolerance=2)(image, meta=meta)
        # print(idx, text)
        image = Image.fromarray(self.img_aug(np.array(image)))
        image = image.convert('RGB')
        # image = self.resize_and_pad_to_model_size(image)
        # encoded_text = [self.char2idx[self.SOS]]+[self.char2idx.get(i, self.OOV_idx) for i in text]+[self.char2idx[self.EOS]]
        return image, text, meta

    def __len__(self):
        return len(self.words)


class MyDataset(Dataset):

    def __init__(self, data_config, aff_aug, img_aug):
        # self.img_transform = self.compose_aug(augmentation_config)
        self.datagen = DataGenerator(
            **data_config, aff_aug=aff_aug, img_aug=img_aug)

    def compose_aug(self, augmentation_config):
        transforms = [aug["transform"](**aug["args"])
                      for aug in augmentation_config["augmentations"]]
        probs = [aug["prob"] for aug in augmentation_config["augmentations"]]

        return Compose(transforms, probs)

    def __getitem__(self, idx):
        image, text, meta = self.datagen[idx]
        image = self.datagen.resize_and_pad_to_model_size(image)
        text = self.datagen.encode_text(text)
        text = torch.LongTensor(text)
        image = ToTensor()(image)
        return (image-image.mean())/255, text, len(text)

    def __call__(self, batch_size, shuffle, pin_memory=True, n_workers=8):
        return DataLoader(self, batch_size=batch_size, shuffle=shuffle,
                          pin_memory=pin_memory, num_workers=n_workers,
                          collate_fn=self.collate_fn,
                          )

    def __len__(self):
        return len(self.datagen)


    def collate_fn(self, data):
        # print(len(data))
        images, texts, text_lens = zip(*data)
        images = torch.stack(images)
        texts = torch.cat(texts).unsqueeze(0)
        # texts = pad_sequence(texts, batch_first=True, padding_value=self.datagen.char2idx[self.datagen.EOS])
        text_lens = torch.LongTensor(text_lens)

        return images, texts, text_lens


if __name__ == '__main__':

    from src.config import data_config, augmentation_config

    d = MyDataset(data_config, augmentation_config)
    for img, text, text_len in d(batch_size=1, shuffle=False):
        print(text, text_len)
        break
    # print(d.datagen.letters)
    # image.save(f"output/{i:0>3}.png")
    # print(text)
    # print("".join([d.datagen.idx2char[i] for i in text]))
