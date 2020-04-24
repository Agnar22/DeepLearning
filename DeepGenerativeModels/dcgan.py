import tensorflow as tf
#from tf.keras.layers import Conv2D, Conv2DTranspose, Dense, Flatten, Input, Reshape
#from tf.keras.layers import LeakyReLU, BatchNormalization, Lambda
#from tf.keras.regularizers import l2
#from tf.keras.constraints import max_norm, Constraint
#from tf.keras.losses import binary_crossentropy, mse
#from tf.keras.models import Model, Sequential
#from tf.keras.optimizers import Adam, SGD
#from tf.keras import backend as K
import main
from verification_net import VerificationNet
import numpy as np
import math

#class ClipConstraint(Constraint):
#
#    def __init__(self, clip_value):
#        self.clip_value=clip_value
#
#    def __call__(self, weights):
#        return K.clip(weights, -self.clip_value, self.clip_value)
#    
#    def get_config(self):
#        return {'clip_value':self.clip_value}

def create_generator(z_size, colors=False):
    generator = tf.keras.models.Sequential()
    generator.add(tf.keras.layers.Dense(3*3*384, input_shape = z_size, activation='relu'))
    generator.add(tf.keras.layers.Reshape((3, 3, 384)))
    generator.add(tf.keras.layers.Conv2DTranspose(192, 5, strides=1, padding='valid', activation='relu', kernel_initializer='glorot_normal'))
    generator.add(tf.keras.layers.BatchNormalization())
    generator.add(tf.keras.layers.Conv2DTranspose(96, 5, strides=2, padding='same', activation='relu', kernel_initializer='glorot_normal'))
    generator.add(tf.keras.layers.BatchNormalization())
    generator.add(tf.keras.layers.Conv2DTranspose(1, 5, strides=2, activation='sigmoid', padding='same'))
    return generator

def create_discriminator(z_size, colors=False):
    discriminator = tf.keras.models.Sequential()
    discriminator.add(tf.keras.layers.Conv2D(64, (3, 3), input_shape = (28, 28, 3 if colors else 1), strides=2, use_bias=True,
        padding="same", kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    discriminator.add(tf.keras.layers.LeakyReLU(alpha=0.2))
    discriminator.add(tf.keras.layers.BatchNormalization())
    discriminator.add(tf.keras.layers.Conv2D(128, (3, 3), strides=2, use_bias=True, padding="same", kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    discriminator.add(tf.keras.layers.LeakyReLU(alpha=0.2))
    discriminator.add(tf.keras.layers.BatchNormalization())
    discriminator.add(tf.keras.layers.Conv2D(256, (3, 3), strides=2, use_bias=True, padding="same", kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    discriminator.add(tf.keras.layers.LeakyReLU(alpha=0.2))
    discriminator.add(tf.keras.layers.BatchNormalization(beta_regularizer=tf.keras.regularizers.l2(0.01), gamma_regularizer=tf.keras.regularizers.l2(0.001)))
    discriminator.add(tf.keras.layers.Flatten())
    discriminator.add(tf.keras.layers.Dense(1, activation='sigmoid', use_bias=True, kernel_regularizer=tf.keras.regularizers.l2(0.001)))
    discriminator.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0002, beta_1=0.5), loss='binary_crossentropy')
    return discriminator

def create_gan(generator, discriminator, z_size, colors=False):
    #discriminator.trainable=False
    for layer in discriminator.layers:
       layer.trainable=False
    gan = tf.keras.models.Sequential()
    gan.add(generator)
    gan.add(discriminator)
    #gan.layers[1].trainable=False
    gan.compile(optimizer=tf.keras.optimizers.Adam(lr=0.0002, beta_1=0.5), loss='binary_crossentropy')

    #discriminator.trainable=True
    generator.summary()
    discriminator.summary()
    gan.summary()

    print(gan.trainable_weights)
    print(discriminator.trainable_weights)
    return gan

def set_trainable(model, trainable=False):
    for layer in model.layers:
        layer.trainable = trainable

def wasserstein_loss(true, pred):
    return -K.mean(true * pred)

def fit(gan, generator, discriminator, gen, batch_size=64, epochs=10):
    x, _ = gen.get_full_data_set(training=True)
    noise_dim = generator.layers[0].input_shape[-1]
    gen_loss = []
    disc_loss = []

    _, y = gen.get_random_batch(batch_size=9)
    gen.plot_example(generator.predict(np.random.normal(size=(9, noise_dim))), y)

    generator_target = np.array([[1] for _ in range(batch_size)])

    for epoch in range(epochs):
        print("Epoch {0}/{1}".format(epoch + 1, epochs))
        temp_gen_loss = 0
        temp_disc_loss = 0

        for batch_num in range(math.ceil(2 * x.shape[0] / batch_size)):
            batch_from = batch_num * batch_size // 2
            batch_to = min(batch_from + batch_size // 2, x.shape[0])

            gaussian_noise = np.random.normal(size=(batch_size, noise_dim))
            generated_batch = generator.predict(gaussian_noise)
            pred = gan.predict(gaussian_noise).mean()
            for layer in discriminator.layers:
                  layer.trainable=False
            #discriminator.trainable=False
            #print(discriminator.trainable_weights)
            if pred < 0.8:
                for _ in range(1+batch_num%2):
                    temp_gen_loss+=gan.train_on_batch(
                        gaussian_noise, generator_target
                    )
            #discriminator.trainable=True
            for layer in discriminator.layers:
                  layer.trainable=True

            discriminator_batch = np.concatenate((x[batch_from:batch_to], generated_batch[:batch_size // 2]),
                                                 axis=0)
            discriminator_target = np.array([[1] if x < batch_to - batch_from else [0]
                                             for x in range(discriminator_batch.shape[0])])
            if pred > 0.35:
                #print(pred)
                #discriminator.trainable=True
                temp_disc_loss +=discriminator.train_on_batch(discriminator_batch, discriminator_target)
            if batch_num % 200 == 0:
                print("Gen_loss: {0:.4f} Disc_loss: {1:.4f}".format(temp_gen_loss / ((batch_num+1)*1),
                                                            temp_disc_loss / ((batch_num+1))
                                                            )
                      )
                print(gan.predict(gaussian_noise).mean())
                print(discriminator.predict(x[batch_from:batch_to]).mean())
                #for num, layer in enumerate(discriminator.layers):
                #     print(layer.name)
                #     for elem in layer.get_weights():
                #        print(elem.mean(), elem.max(), elem.min());

        gen_loss.append(temp_gen_loss / x.shape[0])
        disc_loss.append(temp_disc_loss / x.shape[0])
        verifier = VerificationNet(file_name='./models/mono_float_missing.h5')
        verifier.load_weights()
        main.generate_images(generator, gen, verifier)
        generator.save('mono_float_missing_generator.h5')
        discriminator.save('mono_float_missing_discriminator.h5')

    _, y = gen.get_random_batch(batch_size=9)
    gen.plot_example(generator.predict(np.random.normal(size=(9, noise_dim))), y)

    return gen_loss, disc_loss
