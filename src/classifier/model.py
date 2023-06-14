from keras.models import Model
from keras.layers import Input, Dense, GlobalAveragePooling2D
from blocks import conv_block
import config


def build_model():
    model_input = Input((32, 32, 3))
    conv1 = conv_block(model_input, 64)
    conv2 = conv_block(conv1, 128)
    conv3 = conv_block(conv2, 128)
    conv4 = conv_block(conv3, 256, downsample=False)
    gap = GlobalAveragePooling2D()(conv4)
    out = Dense(units=10, activation='softmax')(gap)

    model = Model(model_input, out)
    model.compile(loss=config.LOSS,
                  optimizer=config.OPTIMIZER,
                  metrics=['accuracy'])
    model.summary()

    return model
