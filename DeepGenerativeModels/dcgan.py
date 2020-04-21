from keras.layers import Conv2D, Conv2DTranspose, Dense, Flatten, Input, Reshape
from keras.layers import LeakyReLU, BatchNormalization, Lambda
from keras.regularizers import l2
from keras.constraints import max_norm, Constraint
from keras.losses import binary_crossentropy, mse
from keras.models import Model, Sequential
from keras.optimizers import Adam, SGD
from keras import backend as K
import main
from verification_net import VerificationNet
import numpy as np
import math
import tensorflow as tf

class ClipConstraint(Constraint):

    def __init__(self, clip_value):
        self.clip_value=clip_value

    def __call__(self, weights):
        return K.clip(weights, -self.clip_value, self.clip_value)
    
    def get_config(self):
        return {'clip_value':self.clip_value}


class DCGan:

    def __init__(self, z_size, colors=False):
        self.generator = None
        self.discriminator = None
        self.real_disc = None
        self.generator_discriminator = None
        self.create_dcgan(z_size, colors=colors)

    def create_dcgan(self, z_size, colors=False):
        generator = Sequential()
        generator.add(Dense(1*1*128, input_shape = z_size, kernel_regularizer=l2(0.001), activation='relu'))
        generator.add(Reshape((1, 1, 128)))
        generator.add(Conv2DTranspose(32, (3, 3), strides=4, use_bias=True, kernel_regularizer=l2(0.001),
                            padding='same'))
        generator.add(LeakyReLU(alpha=0.2))
        generator.add(BatchNormalization(axis=-1))
        generator.add(Conv2DTranspose(64, (4, 4), strides=1, use_bias=True, kernel_regularizer=l2(0.001),
                        padding='valid'))
        generator.add(LeakyReLU(alpha=0.2))
        generator.add(BatchNormalization(axis=-1))
        generator.add(Conv2DTranspose(64, (3, 3), strides=2, use_bias=True, kernel_regularizer=l2(0.001),
                        padding='same'))
        generator.add(LeakyReLU(alpha=0.2))
        generator.add(BatchNormalization(axis=-1))
        generator.add(Conv2DTranspose(3 if colors else 1, (3, 3), strides=2, activation='sigmoid', kernel_regularizer=l2(0.001),
                            use_bias=True, padding='same'))
        generator_input = Input(shape=z_size)
        self.generator = Model(generator_input, generator(generator_input))

        discriminator = Sequential()
        discriminator.add(Conv2D(64, (3, 3), input_shape = (28, 28, 3 if colors else 1), strides=2, use_bias=True,
            padding="same", kernel_regularizer=l2(0.001)))
        discriminator.add(LeakyReLU(alpha=0.2))
        discriminator.add(BatchNormalization())
        discriminator.add(Conv2D(128, (3, 3), strides=2, use_bias=True, padding="same", kernel_regularizer=l2(0.001)))
        discriminator.add(LeakyReLU(alpha=0.2))
        discriminator.add(BatchNormalization())
        discriminator.add(Conv2D(256, (3, 3), strides=2, use_bias=True, padding="same", kernel_regularizer=l2(0.001)))
        discriminator.add(LeakyReLU(alpha=0.2))
        discriminator.add(BatchNormalization(beta_regularizer=l2(0.01), gamma_regularizer=l2(0.001)))
        discriminator.add(Flatten())
        discriminator.add(Dense(1, activation='sigmoid', use_bias=True, kernel_regularizer=l2(0.001)))
        
        discriminator_input = Input(shape=(28, 28, 3 if colors else 1))
        self.discriminator = Model(discriminator_input, discriminator(discriminator_input))
        self.discriminator.compile(optimizer=Adam(lr=0.001), loss=binary_crossentropy)

        #self.discriminator.trainable = False
        z = Input(shape=z_size)
        self.generator_discriminator = Model(z, self.discriminator(self.generator(z)))
        self.generator_discriminator.compile(optimizer=Adam(lr=0.001), loss=binary_crossentropy)
       # self.generator_discriminator = Sequential()
       # self.generator_discriminator.add(generator)
       # self.generator_discriminator.add(discriminator)
       # self.generator_discriminator.layers[-1].trainable=False
        #self.generator_discriminator.compile(optimizer=Adam(lr=0.0001), loss=binary_crossentropy)
       # self.generator_discriminator = Model(generator_input_layer,
       #                                      discriminator(generator(generator_input_layer)),
       #                                      name='generator_discriminator'
       #                                      )

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
        #DCGan.set_trainable(self.discriminator, trainable=True)
        #self.discriminator.compile(optimizer=Adam(lr=0.001), loss=binary_crossentropy)

        #self.generator_discriminator.layers[1].trainable = True
       # self.generator_discriminator.layers[2].trainable = False
        #self.generator_discriminator.compile(optimizer=Adam(lr=0.0001), loss=binary_crossentropy)

        x, _ = gen.get_full_data_set(training=True)
        noise_dim = self.generator.layers[0].input_shape[-1]
        gen_loss = []
        disc_loss = []

        _, y = gen.get_random_batch(batch_size=9)
        gen.plot_example(self.generator.predict(np.random.normal(size=(9, noise_dim))), y)

        #for layer in self.discriminator.layers[0].layers:
        #    print(layer.name, layer.trainable)
        gaussian_noise = np.random.normal(size=(batch_size, noise_dim))

        for epoch in range(epochs):
            print("Epoch {0}/{1}".format(epoch + 1, epochs))
            temp_gen_loss = 0
            temp_disc_loss = 0

            for batch_num in range(math.ceil(2 * x.shape[0] / batch_size)):
                batch_from = batch_num * batch_size // 2
                batch_to = min(batch_from + batch_size // 2, x.shape[0])

                generated_batch = self.generator.predict(gaussian_noise)
                generator_target = np.array([[1] for _ in range(batch_size)])
                #self.discriminator.trainable = False

                temp_gen_loss += self.generator_discriminator.train_on_batch(
                    gaussian_noise, generator_target
                )
                print("gen", self.generator.get_weights()[0].max())
                print("dis", self.discriminator.get_weights()[0].max())

                #self.discriminator.trainable = True
                discriminator_batch = np.concatenate((x[batch_from:batch_to], generated_batch[:batch_size // 2]),
                                                     axis=0)
                discriminator_target = np.array([[1] if x < batch_to - batch_from else [0]
                                                 for x in range(discriminator_batch.shape[0])])
                #temp_disc_loss += self.discriminator.train_on_batch(discriminator_batch, discriminator_target)

                if batch_num % 200 == 0:
                    print("Gen_loss: {0:.4f} Disc_loss: {1:.4f}".format(temp_gen_loss / ((batch_num+1)),
                                                                temp_disc_loss / ((batch_num+1))
                                                                )
                          )
                    print(self.generator_discriminator.predict(gaussian_noise).mean())
                    print(self.discriminator.predict(x[batch_from:batch_to]).mean())
                   # print(len(weights))
                   # for num, layer in enumerate(self.real_disc.layers):
                   #      print(layer.name)
                   #      for elem in layer.get_weights():
                   #         print(elem.mean(), elem.max(), elem.min());

            gen_loss.append(temp_gen_loss / x.shape[0])
            disc_loss.append(temp_disc_loss / x.shape[0])
            verifier = VerificationNet(file_name='./models/mono_float_missing.h5')
            verifier.load_weights()
            main.generate_images(self.generator, gen, verifier)
            self.generator.save('mono_float_missing_generator.h5')
            self.discriminator.save('mono_float_missing_discriminator.h5')

        _, y = gen.get_random_batch(batch_size=9)
        gen.plot_example(self.generator.predict(np.random.normal(size=(9, noise_dim))), y)

        return gen_loss, disc_loss
