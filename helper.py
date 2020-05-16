import cv2
import numpy as np
import os


class Helper(object):
    def __init__(self):
        pass

    def load_image(self, path="./dataset/vimeo_triplet/", start=1, end=1):
        # expect path: "./dataset/vimeo_triplet/"
        if not(path.isdir(path)):
            print("ERROR: PATH COULD NOT FOUND!")
            exit(0)
        path += "sequences/00001/{:0>4d}/im{}.png"
        imgs = []
        for set_num in range(start, end):
            img_set = []
            for image_num in range(1, 4):
                if not(path.isFile(path.format(set_num, image_num))):
                    print("ERROR: FILE COULD NOT FOUND! " + path.format(set_num,image_num))
                    exit(0)
                img = cv2.imread(path.format(set_num, image_num))
                img_set.append(img)
            imgs.append(img_set)
        cv2.destroyAllWindows()
        return np.asarray(imgs)

    def save_image(self, path="./output/", index_start=1, images=[]):
        # expect image shape: (1,1,256,448,3)
        if not(path.isdir(path)):
            os.mkdir(path)
            print("Creating save image path:" + path)
        filename = index_start
        path += "{}.png"
        for img_set in images:
            cv2.imwrite(path.format(filename), img_set[0])
            filename += 1
        cv2.destroyAllWindows()

    def plot_image(self, image):
        cv2.imshow('', image)
        cv2.destroyAllWindows()
