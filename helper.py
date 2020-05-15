import cv2
import numpy as np

class helper:
    img = cv2.imread('dataset/vimeo_triplet/sequences/00001/0001/im1.png')
    print(img.shape)  # (h,w,c)
    (2, 3, 256, 448, 3)
    def load_image(self, path, start, end):
        
        return 1