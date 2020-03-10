import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, Dense, Flatten, Reshape, MaxPooling2D
from tensorflow.keras import Model
from tensorflow.keras.optimizers import SGD
import cv2
import numpy as np


def create_classifier(latent_size=None, encoder=None, input_shape=None, output_shape=None):
    if encoder == None:
        input_data = Input(shape=input_shape)
        x = Conv2D(8, (3, 3), activation='relu', use_bias=True)(input_data)
        x = MaxPooling2D()(x)
        x = Conv2D(32, (3, 3), activation='relu', use_bias=True, padding="same")(x)
        x = MaxPooling2D()(x)
        x = Flatten()(x)
        x = Dense(256, activation='relu', use_bias=True)(x)
        x = Dense(latent_size, activation='linear', use_bias=True)(x)
        x = Dense(80, activation='relu', use_bias=True)(x)
        output_layer = Dense(output_shape[0], activation='softmax', use_bias=True)(x)
        return Model(input_data, output_layer, name="semi_supervised_classifier")

    input_layer = Input(shape=input_shape)
    x = encoder(input_layer)
    x = Dense(80, activation='relu', use_bias=True)(x)
    output_layer = Dense(output_shape[0], activation='softmax', use_bias=True)(x)

    return Model(input_layer, output_layer, name="semi_supervised_classifier")


def create_autoencoder(input_shape, params):
    input_data = Input(shape=input_shape)
    x = Conv2D(8, (3, 3), activation='relu', use_bias=True)(input_data)
    x = MaxPooling2D()(x)
    x = Conv2D(32, (3, 3), activation='relu', use_bias=True, padding="same")(x)
    x = MaxPooling2D()(x)
    x = Flatten()(x)
    x = Dense(256, activation='relu', use_bias=True)(x)
    latent_vec = Dense(params['latentSize'], activation='linear', use_bias=True)(x)
    x = Dense(256, activation='relu', use_bias=True)(latent_vec)
    x = Reshape((16, 16, 1))(x)
    x = Conv2DTranspose(32, (5, 5), activation='relu', use_bias=True,
                        padding='valid')(x)
    x = Conv2DTranspose(1, tuple(i - 19 for i in input_shape[:2]), activation='sigmoid', use_bias=True,
                        padding='valid')(x)
    for layer in Model(input_data, x, name='autoencoder').layers:
        print(layer.output_shape)
    return Model(input_data, x, name='autoencoder'), Model(input_data, latent_vec, name='encoder')
