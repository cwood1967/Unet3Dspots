from tensorflow import keras 
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
import data_generator_3d
import time


def get_unet(tile_size, num_channels, num_z):
    input_size = tile_size + (num_z, num_channels)  # + (num_channels,)
    inputs = keras.Input(input_size)
    conv1 = layers.Conv3D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = layers.Conv3D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = layers.MaxPooling3D(pool_size=(2, 2, 1))(conv1)
    conv2 = layers.Conv3D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = layers.Conv3D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = layers.MaxPooling3D(pool_size=(2, 2, 2))(conv2)
    conv3 = layers.Conv3D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = layers.Conv3D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    pool3 = layers.MaxPooling3D(pool_size=(2, 2, 1))(conv3)
    conv4 = layers.Conv3D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = layers.Conv3D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    # drop4 = layers.Dropout(0.5)(conv4)
    pool4 = layers.MaxPooling3D(pool_size=(2, 2, 2))(conv4)

    conv5 = layers.Conv3D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = layers.Conv3D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    # drop5 = layers.Dropout(0.5)(conv5)

    up6 = layers.Conv3D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(layers.UpSampling3D(size=(2,2,2))(conv5))
    merge6 = layers.concatenate([conv4,up6], axis=3)
    conv6 = layers.Conv3D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = layers.Conv3D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

    up7 = layers.Conv3D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(layers.UpSampling3D(size=(2,2,1))(conv6))
    merge7 = layers.concatenate([conv3,up7], axis=3)
    conv7 = layers.Conv3D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = layers.Conv3D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

    up8 = layers.Conv3D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(layers.UpSampling3D(size=(2,2,2))(conv7))
    merge8 = layers.concatenate([conv2,up8], axis=3)
    conv8 = layers.Conv3D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = layers.Conv3D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

    up9 = layers.Conv3D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(layers.UpSampling3D(size=(2,2,1))(conv8))
    merge9 = layers.concatenate([conv1,up9], axis=3)
    conv9 = layers.Conv3D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = layers.Conv3D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = layers.Conv3D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv10 = layers.Conv3D(1, 1, activation='sigmoid')(conv9)

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
tile_size = (512,512)
num_z = 5  # compressed from 50 to 5 in data_generator_3d
# img_size = (1024,1024)

epochs = 30
log_dir = f'./cheng-yi/cnidocyte/logs/3d/{time.time()}/'
model_path = f"{log_dir}/cnidocyte.h5"

# Build model
# model = get_xception_unet(img_size, num_classes, num_channels)
model = get_unet(tile_size, num_channels, num_z)
model.summary()

train_folder = '/n/core/micro/mg2/cyc/rla/cnidocyte_analysis/train/'
train_count = 0
for x in data_generator_3d.generate_tiles_masks_from_rois(train_folder, '-3d.tif', tile_shape=tile_size, no_loop=True, horizontal_flip=False, vertical_flip=False):
    train_count += 1
train_generator = data_generator_3d.generate_tiles_masks_from_rois(train_folder, '-3d.tif', tile_shape=tile_size, horizontal_flip=False, vertical_flip=False)
validation_folder = '/n/core/micro/mg2/cyc/rla/cnidocyte_analysis/validate/'
val_count = 0
for x in data_generator_3d.generate_tiles_masks_from_rois(validation_folder, '-3d.tif', tile_shape=tile_size, no_loop=True, horizontal_flip=False, vertical_flip=False):
    val_count += 1
validation_generator = data_generator_3d.generate_tiles_masks_from_rois(validation_folder, '-3d.tif', tile_shape=tile_size, horizontal_flip=False, vertical_flip=False)

model.compile(optimizer="adam", loss="binary_crossentropy", )
callbacks = [
    keras.callbacks.ModelCheckpoint(model_path, save_best_only=True),
    keras.callbacks.TensorBoard(log_dir=log_dir),
]
model.fit(train_generator, epochs=epochs, steps_per_epoch=train_count, validation_data=validation_generator, validation_steps=val_count, callbacks=callbacks)