import os

from helper import Helper
# from img_transformer import ImgTransformer
from adaConv import adaConv

if __name__ == "__main__":  
    helper = Helper()
    data_path = "./dataset/vimeo_triplet/"
    train_image = helper.load_image(data_path, 1, 2)

    model = adaConv()
    model.ada_conv_train(train_image)