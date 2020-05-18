from __future__ import print_function, division

from keras.layers import Input, Dense, Reshape, Flatten, Conv2DTranspose
from keras.layers import BatchNormalization, Activation
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf

from stacked_mnist import DataMode, StackedMNISTData
from verification_net import VerificationNet
import numpy as np


class DCGAN():
    def __init__(self, color):
        # Input shape
        self.img_shape = (28, 28, 3 if color else 1)
        self.latent_dim = 100

        # Build and compile the discriminator
        self.discriminator = self.create_discriminator(color)
        self.discriminator.compile(loss='binary_crossentropy',
                                   optimizer=Adam(0.0005, 0.5),
                                   metrics=['accuracy'])

        self.generator = self.create_generator(color)

        z = Input(shape=(self.latent_dim,))
        img = self.generator(z)
        self.discriminator.trainable = False
        pred = self.discriminator(img)
        self.combined = Model(z, pred)
        self.combined.compile(loss='binary_crossentropy', optimizer=Adam(0.0001, 0.5))

    def create_generator(self, color):
        # Create the generator.
        model = Sequential()
        model.add(Dense(128 * 7 * 7, activation="relu", input_dim=self.latent_dim))
        model.add(Reshape((7, 7, 128)))
        model.add(Conv2DTranspose(128, 3, padding='same', strides=2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(Conv2DTranspose(256, 3, padding='same', strides=2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Activation("relu"))
        model.add(Conv2D(3 if color else 1, 3, padding='same'))
        model.add(Activation("tanh"))
        model.summary()

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)
        return Model(noise, img)

    def create_discriminator(self, color):
        # Create the discriminator.
        model = Sequential()
        model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=(28, 28, 3 if color else 1), padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv2D(256, kernel_size=3, strides=1, padding="same"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))
        model.summary()

        img = Input(shape=self.img_shape)
        validity = model(img)
        return Model(img, validity)

    def fit(self, gen, filename, epochs=100, verifier=None, batch_size=128, save_interval=50):
        times = 3
        X_train, _ = gen.get_full_data_set(training=True)
        X_train = (X_train - 0.5) * 2

        # Adversarial ground truths.
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]

            # Generate a batch of new images by sampling noise.
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            gen_imgs = self.generator.predict(noise)

            # Train the discriminator.
            d_loss_real = self.discriminator.train_on_batch(imgs, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # Train the generator.
            g_loss = self.combined.train_on_batch(noise, valid)

            print("{0} Disc loss: {1:.3f}, acc.: {2:.3f}% Gen loss: {3:.3f}".format(epoch, d_loss[0], 100 * d_loss[1],
                                                                                    g_loss))
            if d_loss[1] > 0.8:
                times = min(10, times + 1)
            elif d_loss[1] < 0.3:
                times = max(1, times - 1)

            if epoch % save_interval == 0:
                if verifier is not None:
                    predictability = verifier.check_predictability(self.generator.predict(noise))[0]
                    class_coverage = verifier.check_class_coverage(self.generator.predict(noise))
                    print(predictability, class_coverage)
                    self.save_weights("{0]_".format(epoch) + filename)

    def save_weights(self, filename):
        self.discriminator.save('./models_gan/disc_' + filename)
        self.generator.save('./models_gan/gen_' + filename)

    def load_weights(self, filename):
        self.discriminator.load_weights('./models_gan/disc_'+filename)
        self.generator.load_weights('./models_gan/gen_' + filename)


if __name__ == '__main__':
    # Limit gpu usage.
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    set_session(sess)

    dataset = 'mono_float_complete'

    verifier = VerificationNet(file_name='./models/' + dataset + '.h5')
    verifier.load_weights()
    gen = StackedMNISTData(mode=DataMode.MONO_FLOAT_COMPLETE, default_batch_size=9)

    dcgan = DCGAN(True if 'color' in dataset else False)
    dcgan.fit(gen, verifier, epochs=100000, batch_size=32, save_interval=500)
