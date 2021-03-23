from tensorflow import keras 
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
import data_generator
import time

def get_xception_unet(img_size, num_classes, num_channels):
    inputs = keras.Input(shape= img_size + (num_channels,))

    ### [First half of the network: downsampling inputs] ###

    # Entry block
    x = layers.Conv2D(32, 3, strides=2, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    # Blocks 1, 2, 3 are identical apart from the feature depth.
    for filters in [64, 128, 256]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(filters, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    ### [Second half of the network: upsampling inputs] ###

    for filters in [256, 128, 64, 32]:
        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.UpSampling2D(2)(x)

        # Project residual
        residual = layers.UpSampling2D(2)(previous_block_activation)
        residual = layers.Conv2D(filters, 1, padding="same")(residual)
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    # Add a per-pixel classification layer
    outputs = layers.Conv2D(num_classes, num_classes, activation="softmax", padding="same")(x)

    # Define the model
    model = keras.Model(inputs, outputs)
    return model

def get_unet(img_size, num_channels):
    input_size = img_size + (num_channels,)
    inputs = keras.Input(input_size)
    conv1 = layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = layers.Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = layers.Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    # drop4 = layers.Dropout(0.5)(conv4)
    pool4 = layers.MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = layers.Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = layers.Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    # drop5 = layers.Dropout(0.5)(conv5)

    up6 = layers.Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(layers.UpSampling2D(size=(2,2))(conv5))
    merge6 = layers.concatenate([conv4,up6], axis=3)
    conv6 = layers.Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = layers.Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

    up7 = layers.Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(layers.UpSampling2D(size=(2,2))(conv6))
    merge7 = layers.concatenate([conv3,up7], axis=3)
    conv7 = layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = layers.Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

    up8 = layers.Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(layers.UpSampling2D(size=(2,2))(conv7))
    merge8 = layers.concatenate([conv2,up8], axis=3)
    conv8 = layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = layers.Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

    up9 = layers.Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(layers.UpSampling2D(size=(2,2))(conv8))
    merge9 = layers.concatenate([conv1,up9], axis=3)
    conv9 = layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = layers.Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = layers.Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv10 = layers.Conv2D(1, 1, activation='sigmoid')(conv9)

    model = keras.Model(inputs=inputs, outputs=conv10)

    model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

    return model

# Free up RAM in case the model definition cells were run multiple times
keras.backend.clear_session()
keras.backend.set_image_data_format('channels_last')

# Specify number of pixel classes
num_classes = 1  # For binary mask
num_channels = 1

# Set the image size as the size of the mosaic tiles
img_size = (512,512)
# img_size = (1024,1024)

epochs = 50
log_dir = f'./cheng-yi/cnidocyte/logs/{time.time()}/'
model_path = f"{log_dir}/cnidocyte.h5"

# Build model
# model = get_xception_unet(img_size, num_classes, num_channels)
model = get_unet(img_size, num_channels)
model.summary()

train_folder = '/n/core/micro/mg2/cyc/rla/cnidocyte_analysis/train/'
train_count = 0
for x in data_generator.generate_tiles_masks_from_rois(train_folder, 'tif', no_loop=True, horizontal_flip=False, vertical_flip=False):
    train_count += 1
train_generator = data_generator.generate_tiles_masks_from_rois(train_folder, '.tif', horizontal_flip=False, vertical_flip=False)
validation_folder = '/n/core/micro/mg2/cyc/rla/cnidocyte_analysis/validate/'
val_count = 0
for x in data_generator.generate_tiles_masks_from_rois(validation_folder, 'tif', no_loop=True, horizontal_flip=False, vertical_flip=False):
    val_count += 1
validation_generator = data_generator.generate_tiles_masks_from_rois(validation_folder, '.tif', horizontal_flip=False, vertical_flip=False)

model.compile(optimizer="adam", loss="binary_crossentropy", )
callbacks = [
    keras.callbacks.ModelCheckpoint(model_path, save_best_only=True),
    keras.callbacks.TensorBoard(log_dir=log_dir),
]
model.fit(train_generator, epochs=epochs, steps_per_epoch=train_count, validation_data=validation_generator, validation_steps=val_count, callbacks=callbacks)