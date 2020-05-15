import cv2
import numpy as np


class helper:
    def __init__(self):
        pass

    def load_image(self, path="./dataset/vimeo_triplet/", start=1, end=1):
        # path = "./dataset/vimeo_triplet/"
        path += "sequences/00001/{:0>4d}/im{}.png"
        imgs = []
        for set_num in range(start, end):
            set = []
            for image_num in range(1, 4):
                img = cv2.imread(path.format(set_num,image_num))
                set.append(img)
            imgs.append(set)
        cv2.destroyAllWindows()
        return np.asarray(imgs)

    def save_image(self, path="./output/", index_start=1, images=[]):
        filename = index_start
        path += "{}.png"
        for img_set in images:
            cv2.imwrite(path.format(filename), img_set[0])
            filename += 1
        cv2.destroyAllWindows()

    def plot_image(self,image):
        cv2.imshow(image)
