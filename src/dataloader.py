from PIL import Image, ImageFont, ImageDraw
import os
import numpy as np
import random
from augmentations import RandomBgColor
from augmentations import Compose

class DataGenerator:

    def __init__(self, fonts_path, fonts, font_sizes, words_path, length, transforms):

        with open(words_path) as f:
            self.words = f.read()
        self.words = self.words.strip()

        self.words = self.words.replace(
            "\n\n", "\n").replace("  ", " ").split()

        self.fonts_path = fonts_path
        self.fonts = fonts
        self.font_sizes = font_sizes
        self.length = length

        self.transforms = transforms

    def get_random_font(self, meta={}):

        fontFile = random.choice(self.fonts)
        meta['fontFile'] = fontFile
        fontSize = random.randint(*self.font_sizes)
        meta['fontSize'] = fontSize
        # load the font and image
        font = ImageFont.truetype(os.path.join(
            self.fonts_path, fontFile), fontSize, layout_engine=ImageFont.LAYOUT_RAQM)
        return font, meta

    def generate_text(self, length, meta={}):

        # TODO: Add more types of random texts with different probability.
        text = " ".join([random.choice(self.words)
                        for i in range(length)])
        meta["text"] = text
        return text, meta

    def crop_tight(self, image, padding_per=0.1):
        w, h = image.size
        img = np.array(image)[:, :, :3]
        vt, hz = np.where((img < 255).all(axis=-1))

        x0, y0, x1, y1 = (hz.min(), vt.min(), hz.max(), vt.max())
        padding_x = padding_per*(x1-x0)
        padding_y = padding_per*(y1-y0)

        x0 = max(0, x0-padding_x)
        x1 = min(w, x1+padding_x)
        y0 = max(0, y0-padding_y)
        y1 = min(h, y1+padding_y)

        image = image.crop((x0, y0, x1, y1))
        return image

    def draw_image(self, font, text, meta):

        w, h = font.getsize(text, direction='rtl')
        w = int(w+(2*w))  # PIL gives wrong sizes at certain font sizes. just keeeping
        h = int(h+(2*h))  # these large enough to almost always encompass the whole text
        
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



        padding_per = random.random()*0.05
        meta["padding_per"] = padding_per
        


        image = self.crop_tight(image, padding_per)
        return image, meta

    def generate_clean_sample(self):

        meta = {}

        font, meta = self.get_random_font(meta)

        length = random.randint(*self.length)
        meta["length"] = length

        text, meta = self.generate_text(length, meta)

        image, meta = self.draw_image(font, text, meta)

        return image, meta

    def __getitem__(self, idx):
        image, meta = self.generate_clean_sample()
        image, meta = self.transforms(image, meta)
        return image.convert('RGB'), meta  


def compose_aug(augmentation_config):
    transforms = [aug["transform"](**aug["args"]) for aug in augmentation_config]
    probs = [aug["prob"] for aug in augmentation_config]

    return Compose(transforms, probs)

if __name__ == '__main__':

    from config import data_config, augmentation_config
    
    transform = compose_aug(augmentation_config["augmentations"])
    d = DataGenerator(**data_config, transforms=transform)
    for i in range(20):
        image, meta = d[1]
        image.save(f"output/{i:0>3}.png")
    print(meta)
