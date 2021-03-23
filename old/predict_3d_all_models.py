import os 

import matplotlib.pyplot as plt
import numpy as np
import PIL.Image

from tensorflow.keras.models import load_model
import tensorflow.keras.backend as K

# from MightyMosaic import MightyMosaic
from pims import ImageSequence
import utils

import data_generator_3d_tiles as gen
import tifffile
from tqdm import tqdm

'''
Use this script to run predictions with all saved models, saving
a prediction on each validation image. This allows visualization
of the training process and better model selection.
'''

model_dir = '/ssd1/rla/cheng-yi/cnidocyte/logs/3d-biff/sigmoid/1611605210.8540528'
image_dir = '/n/core/micro/asa/fgm/smc/20190919_Screen/DeepLearn/Training/RICHARD/Validate/'
image_extension = '.tif'
tile_size = (512,512,16)
overlap_fraction=(0.5,0.5,0.5)

def predict_all(model_dir, image_dir, image_extension, tile_size, overlap_fraction):
    print('Starting prediction of all saved models...')
    for root, dirs, files in os.walk(model_dir):
        files.sort()
        files = [f for f in files if f.endswith('.h5')]
        for i, f in enumerate(tqdm(files)):
            K.clear_session()
            model_path = os.path.join(model_dir, f)
            model = load_model(model_path)
            # model.summary()
            model_name = '.'.join(f.split('.')[:-1])
            save_dir = os.path.join(model_dir, 'output_images')
            os.makedirs(save_dir, exist_ok=True) 
            for image_tile, mask_tile, image_tile_name in gen.generate_image_mask_tiles(image_dir, '.tif', 
                                                        tile_size=tile_size, overlap_fraction=overlap_fraction, 
                                                        no_loop=True, horizontal_flip=False, vertical_flip=False,
                                                        random_rotation=False, random_shift=False,
                                                        yield_single_file=True, yield_image_name=True):
                pred_3d = model.predict(image_tile)
                # pred_3d = pred_3d[0,:,:,:,:]
                pred_3d = np.moveaxis(pred_3d,1,-1)
                pred_3d = np.moveaxis(pred_3d,1,-1)
                save_path = os.path.join(save_dir, model_name, f'{image_tile_name}.tif')
                tifffile.imsave(save_path, pred_3d)
                if i == 0:
                    # Save mask ONLY on first epoch since it doesn't change
                    save_path = os.path.join(save_dir, 'mask', f'{image_tile_name}_mask.tif')
                    mask_tile = mask_tile.astype(np.float32)
                    # mask_tile = mask_tile[0,:,:,:,:]
                    mask_tile = np.moveaxis(mask_tile,1,-1)
                    mask_tile = np.moveaxis(mask_tile,1,-1)
                    tifffile.imsave(save_path, mask_tile)
        break  # No subdirectories
    K.clear_session()

# predict_all(model_dir, image_dir, image_extension, tile_size, overlap_fraction)