import numpy as np
from keras.utils import to_categorical, img_to_array
from keras.datasets import cifar10


def get_dataset():
    (train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

    train_labels = to_categorical(train_labels)
    test_labels = to_categorical(test_labels)

    # normalize images and set the precision to FP16
    train_images = np.array(train_images).astype(np.float16) / 255.0
    test_images = np.array(test_images).astype(np.float16) / 255.0

    return (train_images, train_labels), (test_images, test_labels)


def preprocess_image(image):
    if image.mode != "RGB":
        image = image.convert("RGB")

    image = image.resize((32, 32))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = image / 255.0
    return image
