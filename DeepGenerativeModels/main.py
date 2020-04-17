from auto_encoder import AutoEncoder
from verification_net import VerificationNet
from stacked_mnist import DataMode, StackedMNISTData
from keras.optimizers import SGD
import numpy as np

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


if __name__ == '__main__':
    import tensorflow as tf
    from keras.backend.tensorflow_backend import set_session
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    set_session(sess)

    dataset = "color_binary_complete"

    gen = StackedMNISTData(mode=DataMode.COLOR_FLOAT_COMPLETE, default_batch_size=9)
    img, cls = gen.get_random_batch(batch_size=9)
    #gen.plot_example(img, cls)

    x, y = gen.get_full_data_set(training=True)
    
    auto_encoder = AutoEncoder((28, 28, 3), 250)
    auto_encoder.auto_encoder.compile(optimizer=SGD(0.01, momentum=0.99), loss="binary_crossentropy")
    auto_encoder.auto_encoder.fit(x, x, batch_size=8, epochs=20)
    auto_encoder.auto_encoder.save(dataset + '.h5')

    #img = np.around(auto_encoder.auto_encoder.predict(img))
    auto_encoder.auto_encoder.load_weights(dataset + '.h5')
    img = auto_encoder.auto_encoder.predict(img)
    #print((np.random.rand(9, 1, 200) - 0.5 ).shape)
    #img = auto_encoder.decoder.predict((np.random.rand(9, 1, 200) - 0.5 ) * 2 )
    #gen.plot_example(img, cls)

    verifier = VerificationNet(file_name = './models/' + dataset + '.h5')
    verifier.train(gen)
    img, cls = gen.get_random_batch(batch_size=2000)
    img = auto_encoder.auto_encoder.predict(img)
    print(verifier.check_predictability(img, cls))
   # x, y = gen.get_random_batch(training=True, batch_size=50000)
   # 
   # auto_encoder = AutoEncoder((28, 28, 1), 200)
   # auto_encoder.auto_encoder.compile(optimizer=SGD(0.01, momentum=0.9), loss="mse")
   # auto_encoder.auto_encoder.fit(x, x, batch_size=8, epochs=15)

   # #img = np.around(auto_encoder.auto_encoder.predict(img))
   # #img = auto_encoder.auto_encoder.predict(img)
   # #gen.plot_example(img, cls)

   # auto_encoder.auto_encoder.save('mono_float_missing.h5')
