import numpy as np
from sklearn.feature_extraction.image import extract_patches_2d
from PIL import Image


class ImgTransformer(object):
    """
    Provide miscellaneous transforming operations on image data

    Attributes:
        debug: A boolean indicate if printing verbose information
    """
    debug = False

    def __init__(self, debug=False):
        self.debug = debug
        if debug:
            print("img_transformer created...")

    def image_padding(self, image, pad_size, mode='constant', constant_values=0):
        """
        padding a RGB image
        :param image:  a numpy array of a RGB image with shape (n, m, 3)
        :param pad_size:  size of the padding around the image
        :param mode:  the mode about how to generate values of the padding pixels
        :param constant_values:  the value used for 'constant' mode
        :return:  a numpy array of a RGB image with shape (n+2*pad_size, m+2*pad_size, 3)
        """
        # npad is a tuple of (i_before, i_after) for each dimension i,
        # here just pad on horizontal and vertical dimension for a RGB image
        npad = ((pad_size, pad_size), (pad_size, pad_size), (0, 0))
        if mode == 'constant':
            return np.pad(image, npad, mode, constant_values=constant_values)
        else:
            return np.pad(image, npad, mode)

    def image_to_block(self, image_sets, b_size=3, pad_en=True):
        """
        Generate array of blocks, each of which centered at every pixel of the image
        :param image_sets:  a numpy array of RGB image sets with shape (set_cnt, image_cnt_per_set, height, width, 3)
        :param b_size:  value for the height and width of one block
        :param pad_en:  switch for padding operation
        :return:  a numpy array of blocks with shape (set_cnt, image_cnt_per_set, b_size, b_size, 3)
        """
        blocks = []
        for image in image_sets:
            if not pad_en:
                blocks.append(extract_patches_2d(image, (b_size, b_size)))
            else:
                padded_image = self.image_padding(image, (b_size - 1) >> 1)
                blocks.append(extract_patches_2d(padded_image, (b_size, b_size)))

        res = np.asarray(blocks)
        res = res.reshape((res.shape[0]*res.shape[1], b_size, b_size, 3))
        return res

    def image_to_patch(self, image_sets, p_size=3, pad_en=True):
        """
        Generate array of patches, each of which centered at every pixel of the image
        :param image:  a numpy array of RGB image sets with shape (set_cnt, image_cnt_per_set, height, width, 3)
        :param p_size:  value for the height and width of one patch
        :param pad_en:  switch for padding operation
        :return:  a numpy array of patches with shape (set_cnt, image_cnt_per_set, p_size, p_size, 3)
        """
        patches = []
        for image in image_sets:
            if not pad_en:
                patches.append(extract_patches_2d(image, (p_size, p_size)))
            else:
                padded_image = self.image_padding(image, (p_size - 1) >> 1)
                patches.append(extract_patches_2d(padded_image, (p_size, p_size)))
                if self.debug:
                    print("padded_image Shape is: ", padded_image.shape)
                    tmp = patches[0]
                    print("tmp Shape is: ", tmp.shape)
                    print("tmp bytes = ", tmp.nbytes)

        res = np.asarray(patches)
        res = res.reshape((res.shape[0]*res.shape[1], p_size, p_size, 3))
        return res

    def concatenate_patch(self, p1, p2, mode='h'):
        """
        Concatenate two numpy arrays of image patches
        :param p1:  patch 1 with shape (height1, width1, 3)
        :param p2:  patch 2 with shape (height2, width2, 3)
        :param mode:  horizontally concatenate patches whose heights are same,
                      or vertically concatenate patches whose widths are same
        :return: a numpy arrays of concatenated bigger patch
        """
        if mode == 'v':
            return np.vstack((p1, p2))
        else:
            return np.hstack((p1, p2))

    def show_block(self, block):
        Image.fromarray(block, mode='RGB').show()

    def show_patch(self, patch):
        Image.fromarray(patch, mode='RGB').show()
