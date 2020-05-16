import numpy as np
from sklearn.feature_extraction.image import extract_patches_2d
from PIL import Image


class ImgTransformer(object):
    debug = False

    def __init__(self, debug=False):
        self.debug = debug
        if debug:
            print("img_transformer created...")

    def image_to_block(self, images, b_size=3):
        return extract_patches_2d(images, (b_size, b_size))

    def image_to_patch(self, images, p_size=3):
        return extract_patches_2d(images, (p_size, p_size))

    def concatenate_patch(self, p1, p2):
        return np.hstack((p1, p2))

    def show_block(self, block):
        Image.fromarray(block, mode='RGB').show()

    def show_patch(self, patch):
        Image.fromarray(patch, mode='RGB').show()
