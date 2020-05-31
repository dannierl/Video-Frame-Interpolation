import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import numpy as np
import keras.backend as K
from keras.models import Model
from keras.layers import Input, Conv2D, BatchNormalization, MaxPooling2D, Softmax, Reshape, Dot
from keras.callbacks import ModelCheckpoint, EarlyStopping

import tensorflow as tf

from img_transformer import ImgTransformer


class adaConv(object): 
    """
    Attributes:
        debug: A boolean indicate if printing verbose information
        rtx_optimizer: A boolean indicate if RTX optimizer is turned on
    """
    debug = False
    rtx_optimizer = True
    b_size = 79
    p_size = 41

    hdf5_path = "./model/test.hdf5"
    json_path = "./model/test.json"

    def __init__(self, debug=False):
        # =================================================
        # Limit GPU memory(VRAM) usage in TensorFlow 2.0
        # https://github.com/tensorflow/tensorflow/issues/34355
        # https://medium.com/@starriet87/tensorflow-2-0-wanna-limit-gpu-memory-10ad474e2528
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                    print("Limit GPU memory ...")
            except RuntimeError as e:
                print(e)

        # =================================================
        if self.rtx_optimizer == True:
            print("RTX optimizer is on ...")
            K.set_epsilon(1e-4) 

        # =================================================
        self.debug = debug
        if debug:
            print("ada_conv initiated...")

    def ada_conv_predict(self, images, mode = 'default'):
        img_transformer = ImgTransformer()

        # =================================================
        R1 = img_transformer.image_to_block(images[:, 0], self.b_size)
        R2 = img_transformer.image_to_block(images[:, 2], self.b_size)

        R = np.concatenate((R1, R2), axis=3)
        del R1
        del R2

        P1 = img_transformer.image_to_patch(images[:, 0], self.p_size)
        P2 = img_transformer.image_to_patch(images[:, 2], self.p_size)
        P = np.concatenate((P1, P2), axis=2)
        del P1
        del P2
        P = P.reshape((-1, self.p_size*self.p_size*2, 3))
        # =================================================

        from keras.models import model_from_json
        json_file = open(self.json_path, 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        pred_model = model_from_json(loaded_model_json)

        # Load weights from model
        pred_model.load_weights(self.hdf5_path)
        print("Loaded model from " + self.hdf5_path)
        
        # evaluate loaded model on test data
        opt = tf.keras.optimizers.Adam()
        if self.rtx_optimizer == True:
            opt = tf.train.experimental.enable_mixed_precision_graph_rewrite(opt)
        pred_model.compile(optimizer=opt, loss='mse', metrics=['acc'])

        predicted_frames = pred_model.predict([R, P])

        predicted_frames = predicted_frames.reshape(images[0][0].shape)

        return predicted_frames


    def ada_conv_train(self, images, mode = 'default'):
        img_transformer = ImgTransformer()

        # =================================================
        print(images.shape)
        print(images[:, 0].shape)

        R1 = img_transformer.image_to_block(images[:, 0], self.b_size)
        R2 = img_transformer.image_to_block(images[:, 2], self.b_size)

        R = np.concatenate((R1, R2), axis=3)
        del R1
        del R2

        P1 = img_transformer.image_to_patch(images[:, 0], self.p_size)
        P2 = img_transformer.image_to_patch(images[:, 2], self.p_size)
        P = np.concatenate((P1, P2), axis=2)
        del P1
        del P2
        P = P.reshape((-1, self.p_size*self.p_size*2, 3))

        I = images[:, 1].reshape((-1, 3))

        # =================================================
        print(R.shape, R.dtype)
        print(P.shape, P.dtype)
        print(I.shape, I.dtype)

        print("VVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVVV")
        gb = 1024 * 1024 *1024

        print("images_mem ", images.nbytes/gb)
        print("R_mem ", R.nbytes/gb)
        print("P_mem ", P.nbytes/gb)

        total = (R.nbytes + P.nbytes)/gb
        print("total ", total)
        print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")


        block_input = Input(shape = (self.b_size, self.b_size, 6))
        
        x = BatchNormalization()(Conv2D(32, kernel_size=(7, 7), strides = (1, 1), activation='relu')(block_input))
        x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(x)
        x = BatchNormalization()(Conv2D(64, kernel_size=(5, 5), strides = (1, 1), activation='relu')(x))
        x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(x)
        x = BatchNormalization()(Conv2D(128, kernel_size=(5, 5), strides = (1, 1), activation='relu')(x))
        x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid')(x)
        x = BatchNormalization()(Conv2D(256, kernel_size=(3, 3), strides = (1, 1), activation='relu')(x))
        x = Conv2D(2048, kernel_size=(4, 4), strides = (1, 1), activation='relu')(x)
        x = Conv2D(3362, kernel_size=(1, 1), strides = (1, 1), activation='relu')(x)
        x = Softmax()(x)
        x = Reshape((3362, 1))(x)

        # =================================================
        patch_input = Input(shape = (3362, 3))

        y = Dot(axes = 1)([x, patch_input])
        y = Reshape((3,))(y)
        
        pred_model = Model(inputs = [block_input, patch_input], outputs = y)
        pred_model.summary()
        # =================================================
        # Load weights from model
        if mode == 'default':
            pred_model.load_weights(self.hdf5_path)
            print("Reloaded model from " + self.hdf5_path)

        opt = tf.keras.optimizers.Adamax()
        if self.rtx_optimizer == True:
            opt = tf.train.experimental.enable_mixed_precision_graph_rewrite(opt)
        pred_model.compile(optimizer=opt, loss='mse')
        
        # =================================================

        # Save model to json file 
        model_json = pred_model.to_json()
        with open(self.json_path, "w") as json_file:
            json_file.write(model_json)

        # Add early stopping callback
        earlystop = EarlyStopping(monitor='val_loss', min_delta=1.0, \
                                patience=30, \
                                verbose=2, mode='min', \
                                baseline=None, restore_best_weights=True)                    
        # Add modelcheckpoint callback and save model file
        checkpointer = ModelCheckpoint(filepath=self.hdf5_path, \
                                    monitor='val_loss',save_best_only=True)
        callbacks_list = [earlystop, checkpointer]

        pred_model.fit([R, P], I, batch_size=128, epochs=1000, verbose=2, validation_split=0.95, callbacks=callbacks_list)
        # ===================================================