from keras.layers import Dense, Conv2D, BatchNormalization
from keras.layers import LeakyReLU, GlobalAveragePooling2D, Reshape, ReLU, add, multiply


def squeeze_and_excitation_block(input_block, filters, ratio=16):
    x = GlobalAveragePooling2D()(input_block)
    x = Dense(filters // ratio, activation='relu')(x)
    x = Dense(filters, activation='sigmoid')(x)
    x = Reshape((1, 1, filters))(x)

    return multiply([input_block, x])


def conv_block(block_input,
               filters,
               kernel_size=3,
               downsample=True):
    conv = Conv2D(filters, kernel_size,
                  strides=2 if not downsample else 1,
                  padding='same')(block_input)
    batch_norm = BatchNormalization()(conv)
    out = LeakyReLU(0.2)(batch_norm)

    return out
