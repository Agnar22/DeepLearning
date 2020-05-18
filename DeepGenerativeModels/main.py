import auto_encoder
import dcgan
from verification_net import VerificationNet
from stacked_mnist import DataMode, StackedMNISTData

import numpy as np
import json

import tensorflow as tf
from keras.backend.tensorflow_backend import set_session


# # # # # Getting the data # # # # #
# Instantiate the stacked mnist class with a data mode
# - MONO/COLOR
# - BINARY/FLOAT
# - COMPLETE/MISSING (no 8 in training

# # # # # Evaluation of a deep generative model # # # # #
# Verification net: classifier capable of classifying generated data
# - Confidence in most confident class: check_predictability
# - Check accuracy: check_predictability
# - Check mode collapse: check_class_coverage

# # # # # Create auto-encoder # # # # #
# # In general:
# MNIST & STACKED-MNIST
# Stride > 1, conv & transposed conv
# accuracy of 80% with tolerance of 0.8
# # As a generator:
# Generate random vectors as input to decoder
# # As an anomaly detector
# Train with one class missing
# Calculate reconstruction loss for test-data
# Plot most anomalous images


# # # # # Variational auto-encoder # # # # #
# # As a generator:
# (Same as auto-encoder)
# # As an anomaly detector
# (Same as auto-encoder)


# # # # # Generative Adverserial Network # # # # #
# # As a generator:
# Show generated images, check image quality and coverage (as above)


# TIPS:
# - Tensorflow v.1.14.0 w/Keras v.2.2.4 (import keras)

def read_json(filepath):
    with open(filepath) as f:
        return json.load(f)


def get_data_mode(data_mode):
    data_modes = [
        DataMode.MONO_FLOAT_COMPLETE,
        DataMode.MONO_FLOAT_MISSING,
        DataMode.MONO_BINARY_COMPLETE,
        DataMode.MONO_BINARY_MISSING,
        DataMode.COLOR_FLOAT_COMPLETE,
        DataMode.COLOR_FLOAT_MISSING,
        DataMode.COLOR_BINARY_COMPLETE,
        DataMode.COLOR_BINARY_MISSING
    ]

    data_mode_name = [
        'mono_float_complete',
        'mono_float_missing',
        'mono_binary_complete',
        'mono_binary_missing',
        'color_float_complete',
        'color_float_missing',
        'color_binary_complete',
        'color_binary_missing'
    ]

    return data_modes[data_mode_name.index(data_mode)]


def create_model(model_name, latent_size, color=False, binary=False):
    if model_name == 'ae':
        ae = auto_encoder.AE((28, 28, 3 if color else 1), latent_size, variational=False, binary=binary)
        return ae, ae.encoder, ae.decoder
    elif model_name == 'vae':
        vae = auto_encoder.AE((28, 28, 3 if color else 1), latent_size, variational=True)
        return vae, vae.encoder, vae.decoder
    elif model_name == 'dcgan':
        gan = dcgan.DCGAN(color)
        return gan, gan.discriminator, gan.generator


def reconstruct_images(auto_encoder, gen, verifier):
    # img, cls = gen.get_full_data_set(training=False)
    img, cls = gen.get_random_batch(training=False, batch_size=1000)
    rec_img = auto_encoder.predict(img)

    predictability, accuracy = verifier.check_predictability(rec_img, correct_labels=cls)
    print(f"Reconstruction predictability is {predictability:.3f} and accuracy is {accuracy:.3f}.")

    img_b, cls_b = gen.get_random_batch(training=False, batch_size=9)
    gen.plot_example(images=img_b, labels=cls_b)
    rec_img_b = auto_encoder.predict(img_b)
    gen.plot_example(images=rec_img_b, labels=cls_b)
    return predictability, accuracy, rec_img


def generate_images(decoder, gen, verification):
    latent_shape = decoder.layers[0].input_shape[-1]
    rand_latent_vec = np.random.normal(0, 1.1, 1000 * latent_shape).reshape(1000, latent_shape)
    gen_img = decoder.predict(rand_latent_vec)
    _, rand_cls = gen.get_random_batch(batch_size=9)
    gen.plot_example(images=gen_img[:9, :, :], labels=rand_cls)

    predictability, _ = verification.check_predictability(gen_img)
    coverage = verification.check_class_coverage(gen_img)
    print("Predictability is {0:.3f} and coverage is {1:.3f}".format(predictability, coverage))

    return predictability, gen_img


def cross_entropy(targets, predictions, epsilon=1 - 12):
    predictions = np.clip(predictions, epsilon, 1. - epsilon)
    N = predictions.shape[0]
    ce = -(np.sum(targets * np.log(predictions + 1e-9))+np.sum((1-targets) * np.log(1-predictions + 1e-9))) / N
    return ce


def anomaly_detection(model, gen, vae=False, rec_loss_func=lambda x, y: np.mean((x - y) ** 2, axis=(1, 2, 3))):
    if vae:
        num_samples = 1000
        decoder = model.layers[-1]
        decoder.compile(optimizer='adam', loss='binary_crossentropy')
        latent_shape = decoder.layers[0].input_shape[-1]

        x, cls = gen.get_random_batch(training=False, batch_size=300)
        rand_latent_vec = np.random.normal(0, 1, num_samples * latent_shape).reshape(num_samples, latent_shape)
        noise_pred = decoder.predict(rand_latent_vec)

        rec_loss = []
        for img_num in range(x.shape[0]):
            stacked_img = np.array([x[img_num] for _ in range(num_samples)])
            loss = cross_entropy(stacked_img.flatten(), noise_pred.flatten())
            rec_loss.append(loss)
        rec_loss = np.array(rec_loss)
        rec_img = model.predict(x)
    else:
        x, cls = gen.get_full_data_set(training=False)
        rec_img = model.predict(x)
        rec_loss = rec_loss_func(rec_img, x)
    avg_rec_loss = rec_loss.mean()
    print(f"Reconstruction loss: {avg_rec_loss:.3f}")

    ind = rec_loss.argsort()[-25:]
    gen.plot_example(images=x[ind], labels=cls[ind])
    gen.plot_example(images=rec_img[ind], labels=cls[ind])


def train_all():
    param = read_json('pivotal_parameters.json')

    data_mode_name = [
        'mono_float_complete',
        'mono_float_missing',
        'mono_binary_complete',
        'mono_binary_missing',
        'color_float_complete',
        'color_float_missing',
        'color_binary_complete',
        'color_binary_missing'
    ]

    models = ['vae', 'dcgan']

    for model_name in models:
        for dataset in data_mode_name:
            if model_name == 'ae':
                if 'mono' in dataset:
                    latent_size = 40
                else:
                    latent_size = 80
            else:
                if 'mono' in dataset:
                    latent_size = 20
                else:
                    latent_size = 40
            if model_name == 'ae' and (dataset == 'color_float_missing' or dataset == 'color_binary_missing'):
                continue
            if (dataset == 'color_float_complete' or dataset == 'color_binary_missing'):
                continue

            # Initialize verification net.
            force_learn = param['verification']['force_learn']
            verifier = VerificationNet(file_name='./models/' + dataset + '.h5', force_learn=force_learn)
            model, encoder, decoder = create_model(model_name, latent_size, color='color' in dataset,
                                                   binary='binary' in dataset)
            gen = StackedMNISTData(mode=get_data_mode(dataset), default_batch_size=9)
            if param['verification']['load_weights']:
                verifier.load_weights()
            if force_learn:
                verifier.train(gen)
            try:
                model.load_weights(dataset + '.h5')
            except Exception as e:
                print(e)
            rec_pred, rec_acc, _ = reconstruct_images(model, gen, verifier)
            while rec_pred < 0.805 or rec_acc < 0.805:
                print("Training on dataset {0} with {1}".format(dataset, model_name))
                if param['load_weights']:
                    model.load_weights(dataset + '.h5')
                if param['train']:
                    model.fit(gen, batch_size=128, epochs=20)
                if param['save_weights']:
                    model.save_weights(dataset + '.h5')
                x, _ = gen.get_full_data_set(training=True)

                rec_pred, rec_acc, _ = reconstruct_images(model, gen, verifier)


if __name__ == '__main__':
    # Sizes:
    # AE:
    #  x  - mono_float_complete = 40
    #  x  - mono_float_missing = 40
    #  x  - color_float_complete = 80
    #  x  - color_float_missing = 80
    # VAE:
    #  x  - mono_binary_complete = 20
    #  x  - mono_binary_missing = 20
    #  x  - color_binary_complete = 256
    #  x  - color_binary_missing = 256
    # DCGAN:
    #  x  - mono_float_complete
    #  x  - color_float_complete

    # Limit gpu usage.
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    set_session(sess)

    param = read_json('pivotal_parameters.json')
    dataset = param['dataset']

    # Initialize data generator.
    gen = StackedMNISTData(mode=get_data_mode(dataset), default_batch_size=9)
    x, y = gen.get_full_data_set(training=False)

    # Initialize verification net.
    force_learn = param['verification']['force_learn']
    verifier = VerificationNet(file_name='./models/' + dataset + '.h5', force_learn=force_learn)
    model, encoder, decoder = create_model(param['model'], param['latent_size'], color='color' in dataset,
                                           binary='binary' in dataset)
    gen = StackedMNISTData(mode=get_data_mode(dataset), default_batch_size=9)

    if param['verification']['load_weights']:
        verifier.load_weights()
    if force_learn:
        verifier.train(gen)

    # Initialize model, encoder and decoder.
    model, encoder, decoder = create_model(param['model'], param['latent_size'], color='color' in dataset)

    if param['load_weights']:
        model.load_weights(dataset + '.h5')
    if param['train']:
        model.fit(gen, batch_size=128, epochs=30)
    if param['save_weights']:
        model.save_weights(dataset + '.h5')

    if param['model'] != 'dcgan':
        input("Press enter to reconstruct images.")
        print("Reconstructing images")
        reconstruct_images(model, gen, verifier)

    input("Press enter to generate images.")
    print("Generating images")
    generate_images(decoder, gen, verifier)

    if param['model'] != 'dcgan':
        input("Press enter to do anomaly detection.")
        print("Anomaly detection")
        if param['model'] == 'vae':
            anomaly_detection(model.ae, gen, vae=True)
        else:
            anomaly_detection(model, gen, vae=False)
