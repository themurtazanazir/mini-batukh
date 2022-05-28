
from typing import Any, Dict, AnyStr
from src.augmentations import RandomPerespective, RandomRotate, RandomBgColor,\
    RandomResize

data_config: Dict[str, Any] = dict(
    fonts_path="./fonts/",
    words_path="./data/input_data.txt",
    # words_path="./data/input_data_eng.txt",

    fonts=[
        # "ARIBL0.ttf",
        "ScheherazadeNew-Regular.ttf",
        # "Gulmarg Nastaleeq-8.11.2013.ttf",
    ],

    font_sizes=(10, 61),

    length=(1, 2),

    letters=['ص', '(', 'ژ', 'ِ', 'ۄ', 'ٍ', 'ز', 'ڑ', 'ء', 'ة', '¿', '’', ':', 'س', 'ٛ', 'ق', '۵', 'ٸ', ' ', 'ج', 'غ', '&', 'ێ', '‘', 'م', '۰', '۲', 'ر', '٠', 'پ', '#', 'ً', 'ۯ', 'ع', 'ذ', '۳', 'ا', 'ﺅ', 'خ', 'ث', '/', '۱', 'ٚ', 'ط', '"', '٘', '۹', 'ٲ', '۪', 'ح', 'ۂ', 'ہ', 'ٰ', '”', 'ھ', 'ٗ', 'ے', '۶',
             'ٓ', "'", 'ب', 'ٔ', 'ؑ', 'و', '\ufeff', '.', 'ی', 'ٹ', 'ل', 'ۆ', '\x81', 'ۭ', 'ّ', 'ۅ', 'د', ')', 'ؐ', 'ٖ', 'ُ', '>', 'ت', 'ض', '۸', ',', 'ن', 'ئ', 'ظ', 'ں', '،', 'ؓ', 'ڈ', '۔', 'ف', '۴', 'گ', '\xad', 'ؒ', 'ۍ', 'ـ', '؟', 'ک', '٭', 'آ', 'ٮ', '\u200e', 'ٕ', '̡', '!', 'ش', 'َ', '۷', 'ؔ', 'چ', '؛', '='],
    
    
    # letters = ['M', 'n', 'a', ' ', 'm', 'i', 's', 'y', 'd', 'e'],
    
    
    # letters = ['k', 'R', 'D', '-', 'K', '$', ';', 'S', 'p', '’', '7', '.', '”', 'z', 'w', '_', 'B', 'Y', 'ê', '3', 'q', '“', 't', '"', 'V', 'M', '#', 'b', 'J', '%', '0', '\n', ')', 's', ':', 'N', 'r', 'a', '2', '8', 'x', 'I', 'A', 'G', 'e', 'E', 'C', 'g', '6', 'O', 'j', 'X', 'v', 'F', '4', 'T', '9', 'y', 'H', 'n', 'h', '1', '(', 'è', '!', '?', ',', 'W', ']', 'æ', 'c', 'é', 'U', 'o', '[', ' ', 'ô', 'f', 'Q', 'l', 'm', '/', 'd', '—', '*', 'L', 'u', 'P', "'", '5', 'i', '‘'],
    
    model_in_height=32,
    model_in_width=400,
)

augmentation_config = dict(
    augmentations=[
        {
            "transform": RandomResize,
            "args": {
                "min_size_per": 0.5,
                "max_size_per": 1.5,
            },
            "prob": 0.8,

        },
        {
            "transform": RandomPerespective,
            "args": {
                "max_change": 10,
            },
            "prob": 0.2
        },
        {
            "transform": RandomRotate,
            "args": {
                "deg": 10,
            },
            "prob": 0.2
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
