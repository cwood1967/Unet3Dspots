import tensorflow as tf
from tensorflow import keras 
from tensorflow.keras import layers, backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
import data_generator_3d_tiles as gen
import time
import predict_3d_all_models
import tifffile
import numpy as np
import os

# Define a leaky relu function
lrelu = layers.Lambda(lambda x: keras.activations.relu(x, alpha=0.2))

# Define l2 regularizer
# regularizer = tf.contrib.layers.l2_regularizer(scale=0.1)
# regularizer = tf.keras.regularizers.L2(l2=0.1)

# Another the custom loss function from tensorflow
def weighted_tf_bc(pos_weight=1000):
    '''
    A custom loss function for imballanced datasets
    pos_weight: Increases the weight of the True positive class
    '''
    def tf_wbce(y_true, y_pred):
        return tf.nn.weighted_cross_entropy_with_logits(
                            y_true,
                            y_pred,
                            pos_weight,
                        )
    return tf_wbce

# Define custom metrics (Matthews Correlation Coefficient)
def matthews_cc(y_true, y_pred):
    '''
    A custom metric that takes both positive and negative classes into account
    y_true:
    y_pred: 
    '''
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos

    y_pos = K.round(K.clip(y_true, 0, 1))
    y_neg = 1 - y_pos

    tp = K.sum(y_pos * y_pred_pos)
    tn = K.sum(y_neg * y_pred_neg)

    fp = K.sum(y_neg * y_pred_pos)
    fn = K.sum(y_pos * y_pred_neg)

    numerator = (tp * tn - fp * fn)
    denominator = K.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    return numerator / (denominator + K.epsilon())

def mcc_loss(y_true, y_pred):
    
    tp = K.sum(K.cast(y_true*y_pred, 'float'), axis=0)
    tn = K.sum(K.cast((1-y_true)*(1-y_pred), 'float'), axis=0)
    fp = K.sum(K.cast((1-y_true)*y_pred, 'float'), axis=0) * 1e2
    fn = K.sum(K.cast(y_true*(1-y_pred), 'float'), axis=0) / 1e2
    
    up = tp*tn - fp*fn
    down = K.sqrt((tp+fp) * (tp+fn) * (tn+fp) * (tn+fn))
    
    mcc = up / (down + K.epsilon())
    mcc = tf.where(tf.math.is_nan(mcc), tf.zeros_like(mcc), mcc)
    
    return 1 - K.mean(mcc)

class ValImageCallback(keras.callbacks.Callback):
    '''
    log_dir: Path to save prediction images
    validation_dir: 
    '''

    def __init__(self, log_dir, validation_dir, image_extension, tile_size, overlap_fraction, channel_zero_index, force_z, save_on_epoch=20, ):
        super(ValImageCallback, self).__init__()
        self.log_dir = log_dir
        self.validation_dir = validation_dir
        self.image_extension = image_extension
        self.tile_size = tile_size
        self.overlap_fraction = overlap_fraction
        self.save_on_epoch = save_on_epoch
        self.channel_zero_index = channel_zero_index
        self.force_z = force_z

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.save_on_epoch == 0:
            save_dir = os.path.join(self.log_dir, 'output_images')
            os.makedirs(save_dir, exist_ok=True)
            for image_tile, mask_tile, image_tile_name, num, image_shape in gen.generate_image_mask_tiles(self.validation_dir, self.image_extension, 
                                                            tile_size=self.tile_size, overlap_fraction=self.overlap_fraction, 
                                                            no_loop=True, horizontal_flip=False, vertical_flip=False,
                                                            random_rotation=False, random_shift=False,
                                                            yield_single_file=True, yield_image_name=True,
                                                            channel=self.channel_zero_index, force_z=self.force_z):
                
                pred_3d = self.model.predict(image_tile)
                # pred_3d = pred_3d[0,:,:,:,:]
                pred_3d = np.moveaxis(pred_3d,1,-1)
                pred_3d = np.moveaxis(pred_3d,1,-1)
                save_path = os.path.join(save_dir, str(epoch))
                os.makedirs(save_path,exist_ok=True)
                tifffile.imsave(os.path.join(save_path, f'{image_tile_name}.tif'), pred_3d)
                if epoch == 0:
                    # Save mask ONLY on first epoch since it doesn't change
                    save_path = os.path.join(save_dir,'mask')
                    os.makedirs(save_path,exist_ok=True)
                    mask_tile = mask_tile.astype(np.float32)
                    # mask_tile = mask_tile[0,:,:,:,:]
                    mask_tile = np.moveaxis(mask_tile,1,-1)
                    mask_tile = np.moveaxis(mask_tile,1,-1)
                    tifffile.imsave(os.path.join(save_path, f'{image_tile_name}_mask.tif'), mask_tile)   

def get_unet(tile_size, num_channels, activation=lrelu):  # activation='relu'   activation=lrelu
    filters=32
    input_size = tile_size + (num_channels,)
    inputs = keras.Input(input_size)
    conv1 = layers.Conv3D(filters, (3,3,3), activation=activation, padding='same',)(inputs) # 5,5,5 conv 
    conv1 = layers.Conv3D(filters, (3,3,3), activation=activation, padding='same',)(conv1)
    pool1 = layers.MaxPooling3D(pool_size=(2,2,1),)(conv1)
    conv2 = layers.Conv3D(2*filters, (3,3,3), activation=activation, padding='same',)(pool1)
    conv2 = layers.Conv3D(2*filters, (3,3,3), activation=activation, padding='same',)(conv2)
    pool2 = layers.MaxPooling3D(pool_size=(2,2,2))(conv2)
    conv3 = layers.Conv3D(4*filters, (3,3,3), activation=activation, padding='same',)(pool2)
    conv3 = layers.Conv3D(4*filters, (3,3,3), activation=activation, padding='same',)(conv3)
    pool3 = layers.MaxPooling3D(pool_size=(2,2,1),)(conv3)
    conv4 = layers.Conv3D(8*filters, (3,3,3), activation=activation, padding='same',)(pool3)
    conv4 = layers.Conv3D(8*filters, (3,3,3), activation=activation, padding='same',)(conv4)
    pool4 = layers.MaxPooling3D(pool_size=(2,2,2))(conv4)
    conv5 = layers.Conv3D(16*filters, (3,3,3), activation=activation, padding='same',)(pool4)
    conv5 = layers.Conv3D(16*filters, (3,3,3), activation=activation, padding='same',)(conv5)

    up6 = layers.Conv3D(8*filters, 2, activation=activation, padding='same',)(layers.UpSampling3D(size=(2,2,2))(conv5))
    merge6 = layers.concatenate([conv4,up6], axis=4)
    conv6 = layers.Conv3D(8*filters, (3,3,3), activation=activation, padding='same',)(merge6)
    conv6 = layers.Conv3D(8*filters, (3,3,3), activation=activation, padding='same',)(conv6)

    up7 = layers.Conv3D(4*filters, (2,2,1), activation=activation, padding='same',)(layers.UpSampling3D(size=(2,2,1))(conv6))
    merge7 = layers.concatenate([conv3,up7], axis=4)
    conv7 = layers.Conv3D(4*filters, (3,3,3), activation=activation, padding='same',)(merge7)
    conv7 = layers.Conv3D(4*filters, (3,3,3), activation=activation, padding='same',)(conv7)

    up8 = layers.Conv3D(2*filters, 2, activation=activation, padding='same',)(layers.UpSampling3D(size=(2,2,2))(conv7))
    merge8 = layers.concatenate([conv2,up8], axis=4)
    conv8 = layers.Conv3D(2*filters, (3,3,3), activation=activation, padding='same',)(merge8)
    conv8 = layers.Conv3D(2*filters, (3,3,3), activation=activation, padding='same',)(conv8)

    up9 = layers.Conv3D(filters, (2,2,1), activation=activation, padding='same',)(layers.UpSampling3D(size=(2,2,1))(conv8))
    merge9 = layers.concatenate([conv1,up9], axis=4)
    conv9 = layers.Conv3D(filters, (3,3,3), activation=activation, padding='same',)(merge9)
    conv9 = layers.Conv3D(filters, (3,3,3), activation=activation, padding='same',)(conv9)
    # conv9 = layers.Conv3D(64, (3,3,3), activation=activation, padding='same',)(conv9)
    conv10 = layers.Conv3D(1, 1, activation=activation)(conv9)
    logits = layers.Activation('tanh')(conv10)

    model = keras.Model(inputs=inputs, outputs=logits)

    model.compile(
        optimizer=Adam(learning_rate=1e-4,), 
        loss='mse',  # loss='binary_crossentropy', loss=weighted_tf_bc(), loss='mse', loss=mcc_loss
        metrics=['mse', 'binary_crossentropy', weighted_tf_bc(), matthews_cc]
        )  #  epsilon=0.1,

    return model

# Free up RAM in case the model definition cells were run multiple times
keras.backend.clear_session()
keras.backend.set_image_data_format('channels_last')

# Specify number of pixel classes
num_classes = 1  # For binary mask
num_channels = 1
# Specify single channel (zero-indexed)
channel_zero_index = 0
# Force the z-dimension to a specific number of slices
force_z = 32

# Set the image size as the size of the mosaic tiles
tile_size = (128,128,32)
overlap_fraction=(0.5,0.5,0.5)
image_extension = '.tif'
batch_size=16
# num_z = 5  # compressed from 50 to 5 in data_generator_3d
# img_size = (1024,1024)

epochs = 400
log_dir = f'/ssd1/rla/cheng-yi/cnidocyte/logs/aubrey/{time.time()}'
model_string = '_e{epoch:03d}-l{val_loss:.5f}'
model_path = f"{log_dir}/aubrey_{model_string}.h5"

# Build model
# model = get_xception_unet(img_size, num_classes, num_channels)
model = get_unet(tile_size, num_channels,)

# # Resume training
# model = load_model(
#     '/ssd1/rla/cheng-yi/cnidocyte/logs/3d-biff/sigmoid/tile_128/wbce/1611966858.1799026/biff_e1131-l1.46589.h5', 
#     compile=False
# )
# model.compile(
#         optimizer=Adam(learning_rate=1e-5,), 
#         loss=mcc_loss,  # loss='binary_crossentropy', loss=weighted_tf_bc(), loss='mse', loss=mcc_loss
#         metrics=['mse', 'binary_crossentropy', weighted_tf_bc(), matthews_cc]
#         )  #  epsilon=0.1,

model.summary(line_length=150)

train_folder = '/n/core/micro/asa/auk/smc/20201106_3PO_IMARE-101128/Training/RICHARD/train/'
train_count = 0
print('Counting training batches')
for x in gen.generate_image_mask_tiles(train_folder, image_extension, batch_size, tile_size=tile_size, overlap_fraction=overlap_fraction, no_loop=True, horizontal_flip=False, vertical_flip=False, channel=channel_zero_index, force_z=force_z):
    train_count += 1
print('Loading training images')
train_generator = gen.generate_image_mask_tiles(train_folder, image_extension, batch_size, tile_size=tile_size, overlap_fraction=overlap_fraction, channel=channel_zero_index, force_z=force_z)
validation_folder = '/n/core/micro/asa/auk/smc/20201106_3PO_IMARE-101128/Training/RICHARD/validate/'
val_count = 0
print('Counting validation batches')
for x in gen.generate_image_mask_tiles(validation_folder, image_extension, batch_size, tile_size=tile_size, overlap_fraction=overlap_fraction, no_loop=True, random_rotation=False, random_shift=False, horizontal_flip=False, vertical_flip=False, channel=channel_zero_index, force_z=force_z):
    val_count += 1
print('Loading validation images')
validation_generator = gen.generate_image_mask_tiles(validation_folder, image_extension, batch_size, tile_size=tile_size, overlap_fraction=overlap_fraction, random_rotation=False, random_shift=False, horizontal_flip=False, vertical_flip=False, channel=channel_zero_index, force_z=force_z)

callbacks = [
    keras.callbacks.ModelCheckpoint(
        model_path, 
        save_best_only=True,
        monitor='val_matthews_cc',  # 'val_weight_bc', # val_binary_crossentropy
        mode='max',
    ),
    keras.callbacks.TensorBoard(log_dir=log_dir),
    ValImageCallback(log_dir, validation_folder, image_extension, tile_size, overlap_fraction, save_on_epoch=20, channel_zero_index=channel_zero_index, force_z=force_z),
]
model.fit(
    train_generator, epochs=epochs, steps_per_epoch=train_count, 
    validation_data=validation_generator, validation_steps=val_count, 
    callbacks=callbacks,
)

# Optionally, make predictions on all saved models after training is complete
# predict_3d_all_models.predict_all(log_dir, validation_folder, image_extension, tile_size, overlap_fraction)
