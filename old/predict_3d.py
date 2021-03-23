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

model_path = '/ssd1/rla/cheng-yi/cnidocyte/logs/3d-biff/tile_128/mse/round/norm_max_pixel/32f/lrelu_fix/1613762975.1452513/biff_e204-l0.00280.h5'
image_dir = '/n/core/micro/asa/fgm/smc/20190919_Screen/DeepLearn/Training/RICHARD/TEST/'  # '/n/core/micro/asa/fgm/smc/20190919_Screen/DeepLearn/Training/RICHARD/TEST/'

tile_size = (128,128,8)
overlap_fraction=(0.5,0.5,0.5)
image_extension = '.tif'
batch_size=32

custom_obj = {
    'Lambda': Lambda,
    'tf_wbce': custom_objects.weighted_tf_bc(pos_weight=1000),
    'matthews_cc': custom_objects.matthews_cc
}

def predict(model_path, image_dir, image_extension, tile_size, overlap_fraction):
            model = load_model(model_path, custom_objects=custom_obj)
            # model.summary()
            save_dir = os.path.join(image_dir, 'inference_images')
            os.makedirs(save_dir, exist_ok=True) 
            for image_tile, mask_tile, image_name, image_count, image_shape in gen.generate_image_mask_tiles(image_dir, '.tif', 
                                                        tile_size=tile_size, overlap_fraction=overlap_fraction, 
                                                        no_loop=True, horizontal_flip=False, vertical_flip=False,
                                                        random_rotation=False, random_shift=False,
                                                        yield_single_file=True, yield_image_name=True):
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


predict(model_path, image_dir, image_extension, tile_size, overlap_fraction)