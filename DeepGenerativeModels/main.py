import auto_encoder as AE
from verification_net import VerificationNet
from stacked_mnist import DataMode, StackedMNISTData
from keras.optimizers import SGD
import numpy as np
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


def reconstruct_images(auto_encoder, gen, verifier):
    img, cls = gen.get_full_data_set(training=False)
    rec_img = auto_encoder.predict(img)

    predictability, accuracy = verifier.check_predictability(rec_img, correct_labels=cls)
    print(f"Reconstruction predictability is {predictability:.2f} and accuracy is {accuracy:.2f}.")

    img_b, cls_b = gen.get_random_batch(training=False, batch_size=9)
    gen.plot_example(images=img_b, labels=cls_b)
    rec_img_b = auto_encoder.predict(img_b)
    gen.plot_example(images=rec_img_b, labels=cls_b)
    return rec_img


def generate_images(decoder, gen, verification):
    latent_shape = decoder.layers[0].input_shape[-1]
    rand_latent_vec = np.random.normal(0, 1, 1000 * latent_shape).reshape(1000, latent_shape)
    gen_img = decoder.predict(rand_latent_vec)
    _, rand_cls = gen.get_random_batch(batch_size=9)

    predictability, _ = verification.check_predictability(gen_img)
    print(predictability)
    coverage = verification.check_class_coverage(gen_img)
    print(coverage)

    gen.plot_example(images=gen_img[:9, :, :], labels=rand_cls)


def anomaly_detection(auto_encoder, gen, rec_loss_func=lambda x, y: np.mean((x - y) ** 2, axis=(1, 2, 3))):
    x, cls = gen.get_full_data_set(training=False)
    rec_img = auto_encoder.predict(x)
    rec_loss = rec_loss_func(rec_img, x)
    avg_rec_loss = rec_loss.mean()
    print(f"Reconstruction loss: {avg_rec_loss:.3f}")

    ind = rec_loss.argsort()[:16]
    gen.plot_example(images=x[ind], labels=cls[ind])
    gen.plot_example(images=rec_img[ind], labels=cls[ind])

if __name__ == '__main__':
    # TODO: load pivotal parameters
    # TODO: VAE
    # TODO: DCGAN

    # Set max gpu usage.
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    set_session(sess)

    dataset = "mono_float_missing"

    # Initialize data generator, auto encoder and the verification net.
    gen = StackedMNISTData(mode=DataMode.MONO_FLOAT_MISSING, default_batch_size=9)
    x, y = gen.get_random_batch(training=False, batch_size=20000)
    verifier = VerificationNet(file_name='./models/' + dataset + '.h5')
    vae = AE.VAE((28, 28, x.shape[-1]), 20)
    #auto_encoder.load_weights(dataset + '.h5')
    #auto_encoder.compile(optimizer=SGD(0.01, momentum=0.99), loss=AE.elbo_loss)
    vae.vae.compile(optimizer=SGD(lr=0.0001, momentum=0.99), loss=vae.elbo_loss)
    #print(vae.vae.predict(x))
    vae.vae.fit(x, x, epochs=3, batch_size=8)


    reconstruct_images(vae.vae, gen, verifier)
    generate_images(vae.decoder, gen, verifier)
    anomaly_detection(vae.vae, gen)
