import numpy as np
from keras.optimizers import SGD
from keras.losses import CategoricalCrossentropy
from keras.callbacks import EarlyStopping, LearningRateScheduler
from keras.preprocessing.image import ImageDataGenerator

BATCH_SIZE = 256
EPOCHS = 120
WARMUP_EPOCHS = 20
INITIAL_LR = np.float16(0.02)
OPTIMIZER = SGD(INITIAL_LR)
LOSS = CategoricalCrossentropy(label_smoothing=0.2)


def schedule(epoch, lr):
    # exponential learning rate increase until it reaches the initial_lr value
    if epoch <= WARMUP_EPOCHS:
        return INITIAL_LR * epoch / WARMUP_EPOCHS

    # when warmed up, cosine lr decay is applied
    else:
        cos = np.cos((epoch * np.pi) / EPOCHS)
        return INITIAL_LR * (0.5 * (1 + cos))


EARLY_STOPPING = EarlyStopping(monitor='val_loss', mode='min', patience=20, restore_best_weights=True)
LR_SCHEDULER = LearningRateScheduler(schedule, verbose=1)
TRAIN_DATAGEN = ImageDataGenerator(rotation_range=10,
                                   width_shift_range=0.15,
                                   height_shift_range=0.15,
                                   horizontal_flip=True,
                                   zoom_range=0.2)
