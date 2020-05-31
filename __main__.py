import os

from helper import Helper

from adaConv import adaConv

if __name__ == "__main__":  
    helper = Helper()
    data_path = "./dataset/vimeo_triplet/"
    train_image = helper.load_image(data_path, 1, 3)

    model = adaConv()
    
    # train_image2 = helper.load_image(data_path, 50, 51)
    model.ada_conv_train(train_image, 'new')