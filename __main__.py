import os
import numpy as np

from helper import Helper

from adaConv import adaConv

if __name__ == "__main__":  
    helper = Helper()
    data_path = "./dataset/vimeo_triplet/"
    train_image = helper.load_image(data_path, 1, 501)
    
    val_data_path = "./dataset/BlowingBubbles_416x240_50/"
    val_image = helper.load_imgs(val_data_path, 0, 500)

    model = adaConv()

    step = 10
    for i in range(0, 500-step):
        print("\n__________________________________________________________________________________________________")
        print("Train on ", i + step, " | ", "Validate on ", i)
        model.ada_conv_train(np.expand_dims(train_image[i + 10], axis=0), \
                            np.expand_dims(val_image[i], axis=0))