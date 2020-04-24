from keras.layers import Conv2D, Conv2DTranspose, Dense, Flatten, Input, Reshape
from keras.layers import LeakyReLU, BatchNormalization, Lambda
from keras.optimizers import Adam, SGD
from keras.losses import binary_crossentropy, mse
from keras.models import Model
from keras import backend as K
import tensorflow as tf


def create_auto_encoder(input_shape, latent_size):
    # Creating the encoder
    input_layer_encoder = Input(shape=input_shape)
    x = Conv2D(64, (3, 3), strides=2, use_bias=True, padding="same")(input_layer_encoder)
    x = LeakyReLU(alpha=0.05)(x)
    x = BatchNormalization(axis=-1)(x)
    x = Conv2D(128, (3, 3), strides=2, activation='relu', use_bias=True, padding="same")(x)
    x = LeakyReLU(alpha=0.05)(x)
    x = BatchNormalization(axis=-1)(x)
    x = Conv2D(256, (3, 3), strides=2, activation='relu', use_bias=True, padding="same")(x)
    x = LeakyReLU(alpha=0.05)(x)
    x = BatchNormalization(axis=-1)(x)
    x = Flatten()(x)
    x = Dense(512, use_bias=True)(x)
    x = LeakyReLU(alpha=0.05)(x)
    x = BatchNormalization(axis=-1)(x)
    x = Dense(latent_size, activation='linear', use_bias=True)(x)
    latent_vec = BatchNormalization(axis=-1)(x)
    encoder = Model(input_layer_encoder, latent_vec)

    # Creating the decoder
    input_layer_decoder = Input(shape=encoder.layers[-1].output_shape)
    x = Dense(512, use_bias=True)(input_layer_decoder)
    x = LeakyReLU(alpha=0.05)(x)
    x = BatchNormalization(axis=-1)(x)
    x = Reshape((4, 4, 32))(x)
    x = Conv2DTranspose(256, (4, 4), use_bias=True,
                        padding='valid')(x)
    x = LeakyReLU(alpha=0.05)(x)
    x = BatchNormalization(axis=-1)(x)
    x = Conv2DTranspose(128, (3, 3), strides=2, use_bias=True,
                        padding='same')(x)
    x = LeakyReLU(alpha=0.05)(x)
    x = BatchNormalization(axis=-1)(x)
    x = Conv2DTranspose(input_shape[-1], (3, 3), strides=2, activation='sigmoid',
                        use_bias=True, padding='same')(x)
    decoder = Model(input_layer_decoder, x, name='decoder')

    auto_encoder = Model(input_layer_encoder, decoder(latent_vec), name='auto_encoder')

    auto_encoder.summary()
    for layer in auto_encoder.layers:
        print(layer.output_shape)

    return auto_encoder, encoder, decoder


class AE:
    def __init__(self, input_shape, latent_size, variational=False):
        self.z = None
        self.z_mean = None
        self.z_log_var = None
        self.ae, self.encoder, self.decoder = self.create_auto_encoder(input_shape, latent_size,
                                                                       variational)
        self.weights_dir = 'models_vae' if variational else 'models_ae'

    def create_auto_encoder(self, input_shape, latent_size, variational):
        # Creating the encoder
        input_layer_encoder = Input(shape=input_shape)
        x = Conv2D(128, (3, 3), strides=2, use_bias=True, padding="same")(input_layer_encoder)
        x = LeakyReLU(alpha=0.05)(x)
        x = BatchNormalization(axis=-1)(x)
        x = Conv2D(256, (3, 3), strides=2, use_bias=True, padding="same")(x)
        x = LeakyReLU(alpha=0.05)(x)
        x = BatchNormalization(axis=-1)(x)
        x = Conv2D(256, (3, 3), strides=2, use_bias=True, padding="same")(x)
        x = LeakyReLU(alpha=0.05)(x)
        x = BatchNormalization(axis=-1)(x)
        x = Flatten()(x)
        x = Dense(256, use_bias=True)(x)
        x = LeakyReLU(alpha=0.05)(x)
        x = BatchNormalization(axis=-1)(x)
        if variational:
            self.z_mean = Dense(latent_size)(x)
            #self.z_mean_mod = Model(input_layer_encoder, self.z_mean)
            self.z_log_var = Dense(latent_size)(x)
            #self.z_log_var_mod = Model(input_layer_encoder, self.z_log_var)
            self.z = Lambda(self.sampling, output_shape=(latent_size,))([self.z_mean, self.z_log_var])
        else:
            self.z = Dense(latent_size, activation='linear', use_bias=True)(x)
        encoder = Model(input_layer_encoder, self.z, name="encoder")

        # Creating the decoder
        input_layer_decoder = Input(shape=(latent_size,))
        x = Dense(256, use_bias=True)(input_layer_decoder)
        x = LeakyReLU(alpha=0.05)(x)
        x = BatchNormalization(axis=-1)(x)
        x = Reshape((4, 4, 16))(x)
        x = Conv2DTranspose(256, (4, 4), use_bias=True,
                            padding='valid')(x)
        x = LeakyReLU(alpha=0.05)(x)
        x = BatchNormalization(axis=-1)(x)
        x = Conv2DTranspose(256, (3, 3), strides=2, use_bias=True,
                            padding='same')(x)
        x = LeakyReLU(alpha=0.05)(x)
        x = BatchNormalization(axis=-1)(x)
        x = Conv2DTranspose(input_shape[-1], (3, 3), strides=2, activation='sigmoid',
                            use_bias=True, padding='same')(x)
        decoder = Model(input_layer_decoder, x, name='decoder')

        ae_output = decoder(encoder(input_layer_encoder))
        auto_encoder = Model(input_layer_encoder, ae_output, name='auto_encoder')

        if variational:
            #auto_encoder.compile(optimizer=Adam(lr=0.0005), loss=self.elbo_loss)
            auto_encoder.compile(optimizer=Adam(lr=0.002), loss=self.elbo_loss)
        else:
            auto_encoder.compile(optimizer='adam', loss='binary_crossentropy')

        encoder.summary()
        decoder.summary()
        auto_encoder.summary()
        for layer in auto_encoder.layers:
            print(layer.output_shape)

        return auto_encoder, encoder, decoder

    def load_weights(self, filename):
        self.ae.load_weights('./' + self.weights_dir + '/' + filename)

    def fit(self, gen, batch_size=64, epochs=10):
        x, _ = gen.get_full_data_set(training=True)
        self.ae.fit(x, x, batch_size=batch_size, epochs=epochs)

    def predict(self, x):
        return self.ae.predict(x)

    def save_weights(self, filename):
        self.ae.save('./' + self.weights_dir + '/' + filename)

    def sampling(self, args):
        z_mean, z_log_var = args
        batch = K.shape(z_mean)[0]
        dim = K.int_shape(z_mean)[1]
        epsilon = K.random_normal(shape=(batch, dim), dtype=tf.float32)
        return z_mean + K.exp(z_log_var) * epsilon

    def elbo_loss(self, true, pred):
        # Elbo loss =  binary cross-entropy + KL divergence
        rec_loss = K.mean(binary_crossentropy(true, pred), axis=[1, 2])
        kl_loss = -0.5 * K.mean(1 + self.z_log_var - K.square(self.z_mean) - K.exp(self.z_log_var), axis=[-1])
        return 0.9 * rec_loss + 0.1 * kl_loss
