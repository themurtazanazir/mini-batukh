
from src.augmentations import RandomPerespective, RandomRotate, RandomBgColor

data_config = dict(
    fonts_path="./fonts/",
    words_path="./data/input_data.txt",

    fonts=[
        "ScheherazadeNew-Regular.ttf",
        "Gulmarg Nastaleeq-8.11.2013.ttf",
    ],

    font_sizes=(25, 60),

    length=(2, 6),

    letters=['ص', '(', 'ژ', 'ِ', 'ۄ', 'ٍ', 'ز', 'ڑ', 'ء', 'ة', '¿', '’', ':', 'س', 'ٛ', 'ق', '۵', 'ٸ', ' ', 'ج', 'غ', '&', 'ێ', '‘', 'م', '۰', '۲', 'ر', '٠', 'پ', '#', 'ً', 'ۯ', 'ع', 'ذ', '۳', 'ا', 'ﺅ', 'خ', 'ث', '/', '۱', 'ٚ', 'ط', '"', '٘', '۹', 'ٲ', '۪', 'ح', 'ۂ', 'ہ', 'ٰ', '”', 'ھ', 'ٗ', 'ے', '۶',
             'ٓ', "'", 'ب', 'ٔ', 'ؑ', 'و', '\ufeff', '.', 'ی', 'ٹ', 'ل', 'ۆ', '\x81', 'ۭ', 'ّ', 'ۅ', 'د', ')', 'ؐ', 'ٖ', 'ُ', '>', 'ت', 'ض', '۸', ',', 'ن', 'ئ', 'ظ', 'ں', '،', 'ؓ', 'ڈ', '۔', 'ف', '۴', 'گ', '\xad', 'ؒ', 'ۍ', 'ـ', '؟', 'ک', '٭', 'آ', 'ٮ', '\u200e', 'ٕ', '̡', '!', 'ش', 'َ', '۷', 'ؔ', 'چ', '؛', '=']
)

augmentation_config = dict(
    augmentations=[
        {
            "transform": RandomPerespective,
            "args": {
                "max_change": 25,
            },
            "prob": 0.8
        },
        {
            "transform": RandomRotate,
            "args": {
                "deg": 10,
            },
            "prob": 0.5
        },
        {
            "transform": RandomBgColor,
            "args": {
                "base_color_range": (195, 255),
                "tolerance": 2,
            },
            "prob": 1.0
        },


    ]

)
