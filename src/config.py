
import random
from typing import Any, Dict, AnyStr
from src.augmentations import RandomPad, RandomPerespective, RandomRotate, RandomBgColor,\
    RandomResize

data_config: Dict[str, Any] = dict(
    fonts_path="./fonts/",
    words_path="./data/input_data.txt",
    # words_path="./data/input_data_eng.txt",

    fonts=[
        'NotoNastaliqUrdu-Regular.ttf',
         'ScheherazadeNew-Regular.ttf',
         'Gulmarg Nastaleeq-8.11.2013.ttf',
         'Adobe Arabic Regular.otf',
    ],

    font_sizes=(10, 61),

    length=(1, 6),

    letters=[
        'ا',
        'ب',
        'پ',
        'ت',
        'ٹ',
        'ث',  
        'ج',
        'چ',
        'ح',
        'خ',
        'د',
        'ڈ',
        'ذ',
        'ر',
        'ڑ',
        'ز',
        'ژ',
        'س',
        'ش',
        'ص',
        'ض',
        'ط',
        'ظ',
        'ع', 
        'غ',
        'ف',
        'ق',
        'ک',
        'گ',
        'ل',
        'م',
        'ن',
        'ں',
        'و',
        'ۆ',
        # 'ۄ',
        'ھ',
        'ء',
        'ی',
        # 'ؠ',
        'ے',
        'آ',
         'ِ',
        ' ',
        'ٲ',
        'ٔ',
        'ؑ',
        'ٕ',
        'َ', ],
        
    
    

    
    # letters = ['k', 'R', 'D', '-', 'K', '$', ';', 'S', 'p', '’', '7', '.', '”', 'z', 'w', '_', 'B', 'Y', 'ê', '3', 'q', '“', 't', '"', 'V', 'M', '#', 'b', 'J', '%', '0', '\n', ')', 's', ':', 'N', 'r', 'a', '2', '8', 'x', 'I', 'A', 'G', 'e', 'E', 'C', 'g', '6', 'O', 'j', 'X', 'v', 'F', '4', 'T', '9', 'y', 'H', 'n', 'h', '1', '(', 'è', '!', '?', ',', 'W', ']', 'æ', 'c', 'é', 'U', 'o', '[', ' ', 'ô', 'f', 'Q', 'l', 'm', '/', 'd', '—', '*', 'L', 'u', 'P', "'", '5', 'i', '‘'],
    
    model_in_height=32,
    model_in_width=400,
)

from imgaug import augmenters as iaa

aff_aug = iaa.Sometimes(0.85,
                        then_list=iaa.Sequential(
                            [
                                iaa.Sometimes(0.7, iaa.Affine(scale=(0.75, 0.75), seed=random.randint(0, 1000),
                                                              translate_percent={"x": (-0.12, 0.12), "y": (-0.05, 0.05)}),
                                              seed=random.randint(0, 1000)),
                                iaa.Sometimes(0.3, iaa.Affine(scale=(0.6, 1), seed=random.randint(0, 1000)),
                                              seed=random.randint(0, 1000)),
                                iaa.Sometimes(0.15, iaa.ShearX((-20, 20), seed=random.randint(0, 1000)),
                                              seed=random.randint(0, 1000)),
                                iaa.Sometimes(0.5, iaa.Affine(rotate=(-3, 3), seed=random.randint(0, 1000), fit_output=True),
                                              seed=random.randint(0, 1000))
                            ]))

image_noises = iaa.Sometimes(0.6,
                             then_list=iaa.Sequential([
                                 iaa.Sometimes(0.3,
                                               iaa.OneOf([iaa.GaussianBlur(sigma=(0.4, 0.8), seed=random.randint(0, 1000)),
                                                          iaa.MotionBlur(k=3, seed=random.randint(0, 1000))])
                                               ),
                                 iaa.Sometimes(0.3,
                                               iaa.OneOf(
                                                   [iaa.Salt(p=(0.05, 0.125), seed=random.randint(0, 1000)),
                                                       iaa.Pepper(p=(0.05, 0.125),
                                                                  seed=random.randint(0, 1000)),
                                                    iaa.SaltAndPepper(p=(0.05, 0.125), seed=random.randint(0, 1000))],)),
                                 iaa.Sometimes(0.3, iaa.Dropout(
                                     p=(0.05, 0.125), seed=random.randint(0, 1000)))
                             ]
                             ))


augmentation_config = dict(
    augmentations=[
        {
            "transform": RandomResize,
            "args": {
                "min_size_per": 0.8,
                "max_size_per": 1.5,
            },
            "prob": 0.3,

        },
        {
            "transform": RandomPad,
            "args": {
                "min_pad_size": 0,
                "max_pad_size": 20,
            },
            "prob": 0.8,   
        },
        {
            "transform": RandomPerespective,
            "args": {
                "max_change": 10,
            },
            "prob": 0.4
        },
        {
            "transform": RandomRotate,
            "args": {
                "deg": 10,
            },
            "prob": 0.4
        },
        {
            "transform": RandomBgColor,
            "args": {
                "base_color_range": (195, 255),
                "tolerance": 2,
            },
            "prob": 0.9
        },


    ],


)


model_config: Dict[str, Dict] = dict(

)
