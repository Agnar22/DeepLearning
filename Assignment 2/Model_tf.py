import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, Dense
from tensorflow.keras import Model
from tensorflow.keras.optimizers import SGD
import cv2
import numpy as np


def create_classifier(encoder=None, freeze_encoder=False, input_shape=None):
    if encoder == None:
        print("Creating classifier")

        input_data = Input(shape=(input_shape,))
        x = Dense(80, activation='linear')(input_data)
        x = Dense(80, activation='relu')(x)
        output_layer = Dense(2, activation='softmax')(x)

        return Model(input_data, output_layer, name="semi_supervised_classifier")

    # TODO: implement freezing of weights
    print("Creating supervised classifier")
    input_layer, latent_vec = encoder

    x = Dense(80, activation='relu')(latent_vec)
    output_layer = Dense(2, activation='softmax')(x)

    return Model(input_layer, output_layer, name="semi_supervised_classifier")


def create_autoencoder(input_shape):
    input_data = Input(shape=(input_shape,))
    # model = tf.keras.Sequential()
    latent_vec = Dense(80, activation='linear')(input_data)
    # latent_vec = Conv2D(1, (3, 3), activation='relu', kernel_regularizer='l2')(input_data)
    # x = Conv2DTranspose(1, (3, 3), activation='sigmoid', kernel_regularizer='l2')(latent_vec)
    # model.add(tf.keras.layers.Conv2D(32, (3, 3)))
    x = Dense(784, activation='sigmoid')(latent_vec)

    return Model(input_data, x, name='autoencoder'), (input_data, latent_vec)


if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.reshape(60000, 784) / 255
    autoencoder, encoder = create_autoencoder((784))
    autoencoder.compile(optimizer=SGD(lr=0.1, momentum=0.9), loss='mse')

    image = autoencoder.predict(x_train[0].reshape(1, 784))
    image = np.array(image).reshape(28, 28, 1)
    cv2.imshow("orig", np.array(x_train[0]).reshape(28, 28, 1))
    image = np.array(image).reshape(28, 28, 1)
    cv2.imshow("pred", image)

    print(encoder.predict(x_train[0].reshape(1, 784)).shape)

    for x in range(1):
        autoencoder.fit(x_train, x_train, batch_size=8, epochs=1)

    for x in range(1000):
        image = autoencoder.predict(x_train[x].reshape(1, 784))
        image = np.array(image).reshape(28, 28, 1)
        cv2.imshow("orig", np.array(x_train[x]).reshape(28, 28, 1))
        cv2.imshow("pred", image)
        cv2.waitKey()
