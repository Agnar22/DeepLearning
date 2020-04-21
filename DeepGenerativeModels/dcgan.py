from keras.layers import Conv2D, Conv2DTranspose, Dense, Flatten, Input, Reshape
from keras.layers import LeakyReLU, BatchNormalization, Lambda
from keras.losses import binary_crossentropy, mse
from keras.models import Model, Sequential
from keras.optimizers import Adam
from keras import backend as K
import numpy as np
import math
import tensorflow as tf


class DCGan:

    def __init__(self, z_size, colors=False):
        self.generator = None
        self.discriminator = None
        self.generator_discriminator = None
        self.create_dcgan(z_size, colors=colors)

    def create_dcgan(self, z_size, colors=False):
        generator_input_layer = Input(shape=z_size)
        x = Reshape((1, 1, -1))(generator_input_layer)
        x = Conv2DTranspose(512, (3, 3), strides=4, use_bias=True,
                            padding='same')(x)
        x = LeakyReLU(alpha=0.05)(x)
        x = BatchNormalization(axis=-1)(x)
        x = Conv2DTranspose(256, (4, 4), strides=1, use_bias=True,
                            padding='valid')(x)
        x = LeakyReLU(alpha=0.05)(x)
        x = BatchNormalization(axis=-1)(x)
        x = Conv2DTranspose(128, (3, 3), strides=2, use_bias=True,
                            padding='same')(x)
        x = LeakyReLU(alpha=0.05)(x)
        x = BatchNormalization(axis=-1)(x)
        x = Conv2DTranspose(3 if colors else 1, (3, 3), strides=2, activation='sigmoid',
                            use_bias=True, padding='same')(x)
        generator = Model(generator_input_layer, x, name='generator')
        self.generator = Sequential()
        self.generator.add(generator)

        discriminator_input_layer = Input(shape=(28, 28, 3 if colors else 1))
        x = Conv2D(64, (3, 3), strides=2, use_bias=True, padding="same")(discriminator_input_layer)
        x = LeakyReLU(alpha=0.05)(x)
        x = BatchNormalization(axis=-1)(x)
        x = Conv2D(128, (3, 3), strides=2, use_bias=True, padding="same")(x)
        x = LeakyReLU(alpha=0.05)(x)
        x = BatchNormalization(axis=-1)(x)
        x = Conv2D(256, (3, 3), strides=2, use_bias=True, padding="same")(x)
        x = LeakyReLU(alpha=0.05)(x)
        x = BatchNormalization(axis=-1)(x)
        x = Flatten()(x)
        x = Dense(1, use_bias=True)(x)
        discriminator = Model(discriminator_input_layer, x, name='discriminator')
        self.discriminator = Sequential()
        self.discriminator.add(discriminator)

        self.generator_discriminator = Model(generator_input_layer,
                                             discriminator(generator(generator_input_layer)),
                                             name='generator_discriminator'
                                             )

        generator.summary()
        discriminator.summary()
        self.generator_discriminator.summary()

    @staticmethod
    def set_trainable(model, trainable=False):
        for layer in model.layers:
            layer.trainable = trainable

    @staticmethod
    def wasserstein_loss(true, pred):
        return -K.mean(true * pred)

    def fit(self, gen, batch_size=64, epochs=10):
        DCGan.set_trainable(self.discriminator, trainable=True)
        self.discriminator.compile(optimizer=Adam(lr=0.00004), loss=DCGan.wasserstein_loss)

        self.generator_discriminator.layers[1].trainable = True
        self.generator_discriminator.layers[2].trainable = False
        self.generator_discriminator.compile(optimizer=Adam(lr=0.00004), loss=DCGan.wasserstein_loss)

        x, _ = gen.get_full_data_set(training=True)
        noise_dim = self.generator.layers[0].input_shape[-1]
        gen_loss = []
        disc_loss = []

        _, y = gen.get_random_batch(batch_size=9)
        gen.plot_example(self.generator.predict(np.random.normal(size=(9, noise_dim))), y)

        for epoch in range(epochs):
            print("Epoch {0}/{1}".format(epoch + 1, epochs))
            temp_gen_loss = 0
            temp_disc_loss = 0

            for batch_num in range(math.ceil(2 * x.shape[0] / batch_size)):
                batch_from = batch_num * batch_size // 2
                batch_to = min(batch_from + batch_size // 2, x.shape[0])

                gaussian_noise = np.random.normal(size=(batch_size, noise_dim))
                generated_batch = self.generator.predict(gaussian_noise)
                generator_target = np.array([[1] for _ in range(batch_size)])

                temp_disc_loss += self.generator_discriminator.train_on_batch(
                    gaussian_noise,
                    generator_target,
                )

                discriminator_batch = np.concatenate((x[batch_from:batch_to], generated_batch[:batch_size // 2]),
                                                     axis=0)
                discriminator_target = np.array([[1] if x < batch_to - batch_from else [-1]
                                                 for x in range(discriminator_batch.shape[0])])
                temp_gen_loss += self.discriminator.train_on_batch(discriminator_batch, discriminator_target)

                if batch_num % 50 == 0:
                    print("Gen_loss: {0:.4f} Disc_loss: {1:.4f}".format(temp_gen_loss / ((batch_num+1) * batch_size),
                                                                temp_disc_loss / ((batch_num+1) * batch_size)
                                                                )
                          )
            gen_loss.append(temp_gen_loss / x.shape[0])
            disc_loss.append(temp_disc_loss / x.shape[0])

        _, y = gen.get_random_batch(batch_size=9)
        gen.plot_example(self.generator.predict(np.random.normal(size=(9, noise_dim))), y)

        return gen_loss, disc_loss
