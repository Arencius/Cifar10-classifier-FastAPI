import numpy as np
import utils
import config
from model import build_model

(train_images, train_labels), (test_images, test_labels) = utils.get_dataset()
cifar_classifier = build_model()

history = cifar_classifier.fit(config.TRAIN_DATAGEN.flow(train_images, train_labels, batch_size=config.BATCH_SIZE),
                               epochs=config.EPOCHS,
                               validation_data=(test_images, test_labels),
                               callbacks=[config.EARLY_STOPPING, config.LR_SCHEDULER])

cifar_classifier.evaluate(test_images, test_labels)
cifar_classifier.save('cifar_model.h5')
