import os 

import matplotlib.pyplot as plt
import numpy as np
import PIL.Image

from tensorflow import keras
from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K

# from MightyMosaic import MightyMosaic
from pims import ImageSequence
import utils

import data_generator_3d_tiles as gen
import tifffile
from tqdm import tqdm

# from custom_objects import lrelu
from tensorflow.keras.layers import Lambda
import custom_objects
import yaml

'''
Use this script to run predictions with all saved models, saving
a prediction on each validation image. This allows visualization
of the training process and better model selection.
'''

model_path = '/ssd1/rla/cheng-yi/cnidocyte/logs/aubrey/1615402863.945919/aubrey__e111-l0.00167.h5'
model_dir = '/ssd1/rla/cheng-yi/cnidocyte/logs/aubrey/1615402863.945919/'
image_dir = '/n/core/micro/asa/auk/smc/20201106_3PO_IMARE-101128/Training/RICHARD/TEST/'  # '/n/core/micro/asa/fgm/smc/20190919_Screen/DeepLearn/Training/RICHARD/TEST/'

# Specify single channel (zero-indexed)
channel_zero_index = 0
# Force the z-dimension to a specific number of slices
force_z = 32

# Set the image size as the size of the mosaic tiles
tile_size = (128,128,32)
overlap_fraction=(0.5,0.5,0.5)
image_extension = '.tif'
batch_size=16

custom_obj = {
    'Lambda': Lambda,
    'tf_wbce': custom_objects.weighted_tf_bc(pos_weight=1000),
    'matthews_cc': custom_objects.matthews_cc
}

def predict(model_path, image_dir, image_extension, tile_size, overlap_fraction, channel_zero_index, force_z, model_name=None):
    '''
    Predict a single model. Make sure the values here match exactly the values given durring training.
    model_path: Path to Keras model
    image_dir: Path to image directory
    image_extension: Image extension, ex. '.tif'
    tile_size: Shape of image tiles
    overlap_fraction: List of floats between 0 and 1 representing the fraction of overlap 
        in each dimension where 0 is no overlap and 1 is complete overlap.
        Safe value is 0.5
    channel_zero_index: Integer representing the zero-indexed (0,1,2,...) channel of interest
    force_z: Integer representing the number of z-slices to force each image to have.
    '''
    model = load_model(model_path, custom_objects=custom_obj)
    # model.summary()
    save_dir = os.path.join(image_dir, 'inference_images')
    os.makedirs(save_dir, exist_ok=True) 
    for image_tile, mask_tile, image_name, image_count, image_shape in gen.generate_image_mask_tiles(image_dir, '.tif', 
                                                tile_size=tile_size, overlap_fraction=overlap_fraction, 
                                                no_loop=True, horizontal_flip=False, vertical_flip=False,
                                                random_rotation=False, random_shift=False,
                                                yield_single_file=True, yield_image_name=True,
                                                channel=channel_zero_index, force_z=force_z):
        if model_name:
            save_dir_image = os.path.join(save_dir,model_name,image_name)
        else:
            save_dir_image = os.path.join(save_dir,image_name)
        os.makedirs(save_dir_image, exist_ok=True)
        inference_log_path = os.path.join(save_dir_image, 'inference.yaml')
        if not os.path.exists(inference_log_path):
            image_path = os.path.join(image_dir,f'{image_name}{image_extension}')
            with open(inference_log_path, 'w') as iwf:
                dump = yaml.safe_dump(dict(image_shape=image_shape, image_path=image_path))
                iwf.write(dump)
        image_tile_name = f'{image_name}_{image_count}'
        pred_3d = model.predict(image_tile)
        # pred_3d = pred_3d[0,:,:,:,:]
        pred_3d = np.moveaxis(pred_3d,1,-1)
        pred_3d = np.moveaxis(pred_3d,1,-1)
        save_path = os.path.join(save_dir_image, f'{image_tile_name}.tif')
        tifffile.imsave(save_path, pred_3d)

def predict_all_models(model_dir, image_dir, image_extension, tile_size, overlap_fraction, channel_zero_index, force_z):
    '''
    Predict all models in a directory. Make sure the values here match exactly the values 
    given durring training.
    model_dir: Path to model directory
    image_dir: Path to image directory
    image_extension: Image extension, ex. '.tif'
    tile_size: Shape of image tiles
    overlap_fraction: List of floats between 0 and 1 representing the fraction of overlap 
        in each dimension where 0 is no overlap and 1 is complete overlap.
        Safe value is 0.5
    channel_zero_index: Integer representing the zero-indexed (0,1,2,...) channel of interest
    force_z: Integer representing the number of z-slices to force each image to have.
    '''
    for root, dirs, files in os.walk(model_dir):
        files.sort()
        files = [f for f in files if f.endswith('.h5')]
        for i, f in enumerate(tqdm(files)):
            K.clear_session()
            model_path = os.path.join(model_dir, f)
            # model.summary()
            model_name = '.'.join(f.split('.')[:-1])
            predict(model_path, image_dir, image_extension, tile_size, overlap_fraction, channel_zero_index, force_z, model_name)

predict(model_path, image_dir, image_extension, tile_size, overlap_fraction, channel_zero_index, force_z)

# predict_all_models(model_dir, image_dir, image_extension, tile_size, overlap_fraction, channel_zero_index, force_z)

# for root, dirs, files in os.walk('/ssd1/rla/cheng-yi/cnidocyte/logs/jay_jeff_emma/1614873611_completed'):
#     for f in files:
#         if not f.endswith('.h5'):
#             continue
#         current = os.path.join(root,f)
#         new = current.replace('biff_','jay_jeff_emma_')
#         os.rename(current, new)