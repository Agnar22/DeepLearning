from keras.layers import Conv2D, Conv2DTranspose, Dense, Flatten, Input, MaxPooling2D, Reshape, SeparableConv2D
from keras.layers import LeakyReLU, BatchNormalization
from keras.models import Model


class AutoEncoder:

    def __init__(self, input_shape, latent_size):
        (self.auto_encoder, self.encoder, self.decoder) = self.__create_auto_encoder(input_shape, latent_size)

    def __create_auto_encoder(self, input_shape, latent_size):
        # TODO: convolutions with strides > 1, no maxpooling?
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

        for layer in encoder.layers:
            print(layer.output_shape)

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

        autoencoder = Model(input_layer_encoder, decoder(latent_vec), name='autoencoder')

        for layer in autoencoder.layers:
            print(layer.output_shape)

        return autoencoder, encoder, decoder
