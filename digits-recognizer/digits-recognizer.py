import keras
from keras.models import Sequential, load_model, model_from_json
from keras.layers import Activation, Dense, Dropout, Conv2D, Flatten, MaxPooling2D
from keras.optimizers import Adam, RMSprop, Adamax
from keras.datasets import mnist
from keras import backend


def cnn_mnist_model():
    (mnist_train_images, mnist_train_labels), (mnist_test_images, mnist_test_labels) = mnist.load_data()
    # print(mnist_train_images.shape)  # returns the shape of data so (sample_size, width, height, #channels(if 1 = absent)

    train_images, test_images, input_shape = 0, 0, 0

    if backend.image_data_format() == "channels_first":
        train_images = mnist_train_images.reshape(mnist_train_images.shape[0], 1, 28,
                                                  28)  # reshape takes the sample, #. of channels, width, height wanted
        test_images = mnist_test_images.reshape(mnist_test_images.shape[0], 1, 28, 28)
        input_shape = (1, 28, 28)
    else:
        train_images = mnist_train_images.reshape(mnist_train_images.shape[0], 28, 28,
                                                  1)  # reshape takes the sample, width, height , #. of channels wanted
        test_images = mnist_test_images.reshape(mnist_test_images.shape[0], 28, 28, 1)
        input_shape = (28, 28, 1)

    train_images = train_images.astype('float32')   # makes array of integers floats so division can be performed on it
    test_images = test_images.astype('float32')

    train_images /= 255     # normalize RGB values between 0 to 1 to make it standardized
    test_images /= 255

    train_labels = keras.utils.to_categorical(mnist_train_labels, 10)
    test_labels = keras.utils.to_categorical(mnist_test_labels, 10)

    cnn_model = Sequential([
        Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=input_shape),
        Conv2D(64, kernel_size=(3, 3), activation="relu"),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.5),
        Dense(10, activation="softmax")
    ])

    cnn_model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    cnn_model.fit(train_images, train_labels, batch_size=32, epochs=6, verbose=2, validation_data=(test_images, test_labels))
    # Stop at 6 epochs because performance doesn't get much higher than 99% - feel free to experiment with the epochs

    cnn_model.save("cnn_model_mnist.h5")
    print("Saved model successfully as cnn_model_mnist.h5")

    loaded_model = load_model("cnn_model_mnist.h5")
    score = loaded_model.evaluate(test_images, test_labels)
    print("Saved model has an accuracy of {}% on images it has never seen before.".format(round(score[1]*100), 4))
