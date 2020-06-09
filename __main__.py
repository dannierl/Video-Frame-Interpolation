import os
import gc
import numpy as np

from helper import Helper

from adaConv import adaConv

if __name__ == "__main__":  
    helper = Helper()
    data_path = "./dataset/vimeo_triplet/"
    train_image = helper.load_image(data_path, 1, 501)
    
    val_data_path = "./dataset/BlowingBubbles_416x240_50/"
    val_image = helper.load_imgs(val_data_path, 0, 500)

    step = 10
    for i in range(10, 500 - step, 10):
        print("\n__________________________________________________________________________________________________")
        print("Train on ", i, " | ", "Validate on ", i + step)
        model = adaConv()
        history = model.ada_conv_train(np.expand_dims(train_image[i], axis=0), \
                            np.expand_dims(val_image[i+step], axis=0))
        helper.save_history(history)
        del model
        gc.collect()

