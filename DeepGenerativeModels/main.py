from auto_encoder import AutoEncoder
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
    gen = StackedMNISTData(mode=DataMode.MONO_FLOAT_MISSING, default_batch_size=9)
    img, cls = gen.get_random_batch(batch_size=9)
    gen.plot_example(img, cls)

    x, y = gen.get_random_batch(training=True, batch_size=5000)
    
    auto_encoder = AutoEncoder((28, 28, 1), 200)
    auto_encoder.auto_encoder.compile(optimizer=SGD(0.01, momentum=0.9), loss="mse")
    auto_encoder.auto_encoder.fit(x, x, batch_size=8, epochs=4)

    #img = np.around(auto_encoder.auto_encoder.predict(img))
    img = auto_encoder.auto_encoder.predict(img)
    gen.plot_example(img, cls)
