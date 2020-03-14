from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, Dense, Flatten, Reshape, MaxPooling2D
from tensorflow.keras import Model


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
        return Model(input_data, output_layer, name="supervised_classifier")

    # Creating classifier head
    input_layer = Input(shape=latent_size)
    x = Dense(80, activation='relu', use_bias=True)(input_layer)
    output_layer = Dense(output_shape[0], activation='softmax', use_bias=True)(x)
    classifier_head = Model(input_layer, output_layer, name='classifier_head')

    return Model(encoder.input, classifier_head(encoder.output), name="semi_supervised_classifier")


def create_autoencoder(input_shape, params):
    # Creating the encoder
    input_layer_encoder = Input(shape=input_shape)
    x = Conv2D(8, (3, 3), activation='relu', use_bias=True)(input_layer_encoder)
    x = MaxPooling2D()(x)
    x = Conv2D(32, (3, 3), activation='relu', use_bias=True, padding="same")(x)
    x = MaxPooling2D()(x)
    x = Flatten()(x)
    x = Dense(256, activation='relu', use_bias=True)(x)
    latent_vec = Dense(params['latentSize'], activation='linear', use_bias=True)(x)
    encoder = Model(input_layer_encoder, latent_vec)

    # Creating the decoder
    input_layer_decoder = Input(shape=params['latentSize'])
    x = Dense(256, activation='relu', use_bias=True)(input_layer_decoder)
    x = Reshape((16, 16, 1))(x)
    x = Conv2DTranspose(32, (5, 5), activation='relu', use_bias=True,
                        padding='valid')(x)
    x = Conv2DTranspose(input_shape[-1], tuple(i - 19 for i in input_shape[:2]), activation='sigmoid', use_bias=True,
                        padding='valid')(x)
    decoder = Model(input_layer_decoder, x, name='decoder')

    autoencoder = Model(input_layer_encoder, decoder(latent_vec), name='autoencoder')

    return autoencoder, encoder, decoder
