
from augmentations import RandomPerespective, RandomRotate, RandomBgColor

data_config = dict(
    fonts_path="./fonts/",
    words_path="./data/input_data.txt",

    fonts=[
        "ScheherazadeNew-Regular.ttf",
        "Gulmarg Nastaleeq-8.11.2013.ttf",
    ],

    font_sizes=(10, 60),

    length=(2, 6),

)

augmentation_config = dict(
    augmentations=[
        {
            "transform": RandomPerespective,
            "args": {
                "max_change": 25,
            },
            "prob": 1.0
        },
        {
            "transform": RandomRotate,
            "args": {
                "deg": 10,
            },
            "prob": 1.0
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
