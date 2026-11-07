from pathlib import Path
import cv2
import numpy as np
import datetime

from keras.models import Model
from keras.layers import Input, Conv2D, UpSampling2D, BatchNormalization, Activation
from keras.preprocessing import image


LOAD_WEIGHTS = True
ITERATIONS = 5000
BATCH_SIZE = 1
SEED = 1
IMG_SHAPE = (256, 256, 1)
IMG_HEIGHT, IMG_WIDTH, IMG_CHAN = IMG_SHAPE



def build_model():
    input_tensor = Input((None, None, 1))

    x = Conv2D(64, 11, padding='same')(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(128, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(256, 3, dilation_rate=1, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(256, 3, dilation_rate=2, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(256, 3, dilation_rate=4, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(256, 3, dilation_rate=8, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(128, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(64, 3, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    output_tensor = Conv2D(1, 3, padding='same', activation='sigmoid')(x)

    return Model(input_tensor, output_tensor)


def data_loader():
    input_dir = Path('./Inputs')
    output_dir = Path('./Outputs')

    while True:
        input_images = sorted(input_dir.glob('*.jpg'))

        x_batch = []
        y_batch = []

        for input_file in input_images:
            output_file = output_dir / input_file.name

            if not output_file.exists():
                continue

            # Read broken image
            x_img = cv2.imread(str(input_file), cv2.IMREAD_GRAYSCALE)

            # Read corresponding clean image
            y_img = cv2.imread(str(output_file), cv2.IMREAD_GRAYSCALE)

            if x_img is None or y_img is None:
                continue

            # Resize both images
            x_img = cv2.resize(x_img, (IMG_WIDTH, IMG_HEIGHT))
            y_img = cv2.resize(y_img, (IMG_WIDTH, IMG_HEIGHT))

            # Normalize to [0, 1]
            x_img = x_img.astype(np.float32) / 255.0
            y_img = y_img.astype(np.float32) / 255.0

            # Add channel dimension
            x_batch.append(x_img[..., np.newaxis])
            y_batch.append(y_img[..., np.newaxis])

            if len(x_batch) == BATCH_SIZE:
                yield np.array(x_batch), np.array(y_batch)
                x_batch = []
                y_batch = []

def train():
    model = build_model()
    model.summary()

    if LOAD_WEIGHTS:
        model.load_weights('./weight/model2_200.h5')

    model.compile(loss='MSE', optimizer='Adam')

    data = data_loader()
    start_time = datetime.datetime.now()

    for iteration in range(1, ITERATIONS + 1):

        train_x_batch, train_y_batch = next(data)
        loss = model.train_on_batch(train_x_batch, train_y_batch)

        print('[Time: %s] [Iteration: %d] [Loss: %f]' % (datetime.datetime.now() - start_time, iteration, loss))

        if iteration % 200 == 0:
            model.save('./weight/model2_%d.h5' % iteration)

    model.save('./weight/model2.h5')


if __name__ == "__main__":
    train()