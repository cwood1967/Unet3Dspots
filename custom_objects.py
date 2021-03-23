import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, activations, backend as K
import os
import tifffile
import numpy as np
import data_generator_3d_tiles as gen

# Define a leaky relu function
lrelu = layers.Lambda(lambda x: activations.relu(x, alpha=0.2))

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
    This callback saves validation images every 20 epochs to the given location.
    log_dir: Path to save predicted images
    validation_dir: Path do validation directory
    image_extension: Extension of image files (ex .tif)
    tile_size: The tile x,y,z size
    overlap_fraction: The x,y,z tile overlap fraction between 0 and 1
    save_on_epoch: Saves images after 20 epochs by default.
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
