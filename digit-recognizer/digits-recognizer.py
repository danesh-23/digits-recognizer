import keras
from keras.models import Sequential, load_model, model_from_json
from keras.layers import Activation, Dense, Dropout, Conv2D, Flatten, MaxPooling2D
from keras.optimizers import Adam, RMSprop, Adamax
from keras.datasets import mnist
from keras import backend
import matplotlib.pyplot as plt


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

    cnn_model.fit(train_images, train_labels, batch_size=32, epochs=5, verbose=2, validation_data=(test_images, test_labels))
    # Stop at 5 epochs because performance doesn't get much higher than 99% - feel free to experiment with the epochs

    cnn_model.save("cnn_model_mnist.h5")
    print("Saved model successfully as cnn_model_mnist.h5")

    loaded_model = load_model("/Users/danesh/Desktop/cnn_model_mnist.h5")
    score = loaded_model.evaluate(test_images, test_labels, verbose=0)
    print("Saved model has an accuracy of {}% on images it has never seen before.".format(round(score[1]*100, 4)))

    misclassified_img_prompt = input("Would you like to see some of the images I failed to predict correctly? [Y/N]\n")
    try:
        if misclassified_img_prompt.lower() == "y":
            max_range = int(input("What is the max number of wrong images you would like to see(upper bound)?\n"
                            "Considering I have an accuracy score of {}%, there will be approximately {} misclassified "
                                  "image per 100 images.\nI will be testing my skills on a dataset of 10000 images I"
                                  " have never seen before.\n".format(round(score[1]*100, 4), round((1-score[1])/0.01))))
            check_random_wrongly_classified_images(max_range, loaded_model, test_images, test_labels)
    except:
        print("Invalid input entered.\n")
    print("Thank you for using my services :)\n")


def check_random_wrongly_classified_images(max_range, models, test_img, test_lbl):
    wrong_count = 0
    for index in range(len(test_img)):
        if wrong_count < max_range:
            img = test_img[index, :].reshape(1, 28, 28, 1)
            predicted_num = models.predict(img).argmax()
            true_num = test_lbl[index].argmax()
            if true_num != predicted_num:
                plt.title("Prediction: {}, True value: {}".format(predicted_num, true_num))
                plt.imshow(img.reshape([28, 28]), cmap=plt.get_cmap("gray_r"))
                plt.show()
                wrong_count += 1
