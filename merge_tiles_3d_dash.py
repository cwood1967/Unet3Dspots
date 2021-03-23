import tifffile
import pims
import numpy as np
import data_generator_3d_tiles as data_gen
import os
from tqdm import tqdm
import shutil
import yaml

def merge_and_save(image_dir, merge_shape, tile_size, force_z=None, original_image_path=None, save_to_root=True, channel_axis=1):
    '''
    This function assumes that the inference images only have 3 dimensions z,x,y

    image_dir: The image directory
    merge_shape: The shape of the original image that the tiles will be merged into
    tile_size: Size of the image tiles
    force_z: The z-dimension to force the data into. In this case it is used for opening the 
        original image and forcing it to be the same dimension as the network output.
    save_to_root: If true, save images to the parent directory with the same name as the directory
        If false, create a 'merge' subdirectory and save image as 'merge.tif'
    original_image_path: The path to the original input image
    channel_axis: The original image may have 4 dimensions
    '''

    def resize_z(image, force_z, z_index):
        '''
        Resize the z-dimension of an image given the z-index and the new z-size.
        image: The image to resize
        force_z: The new z-size to enforce. If force_z is greater than the current
        '''
        current_z = image.shape[z_index]
        if force_z > current_z:
            pad_shape = []
            for i, z in enumerate(image.shape):
                if i == z_index:
                    if force_z > z:
                        pad_shape.append((0,force_z-z))
                else:
                    pad_shape.append((0,0))
            image = np.pad(image, pad_shape, 'constant')
        if force_z < current_z:
            x = np.swapaxes(image, 0, z_index)
            x = x[:force_z]
            image = np.swapaxes(x, 0, z_index)
        return image

    merge_image = data_gen.merge_tiles(image_dir, merge_shape, tile_size=tile_size)  # , merge_shape=(512,512,32)
    if original_image_path:
        # Add merge image to last channel of original
        original = tifffile.imread(original_image_path)
        if len(original.shape) == 3 and len(merge_image.shape) == 3:
            # Add new channel axis to both
            if force_z:
                original = resize_z(original, force_z, z_index=0)
            merge_image = np.stack([original, merge_image], axis=channel_axis)
        if len(original.shape)==4 and len(original.shape) > len(merge_image.shape):
            if force_z:
                original = resize_z(original, force_z, z_index=0)
            merge_image = np.expand_dims(merge_image, axis=channel_axis)
            merge_image = np.concatenate([original, merge_image], axis=channel_axis)
    if save_to_root:
        root_dir, name = os.path.split(image_dir)
        merge_dir = os.path.join(image_dir,root_dir)
        merge_path = os.path.join(merge_dir, f'{name}.tif')
    else:
        merge_dir = os.path.join(image_dir,'merge')
        merge_path = os.path.join(merge_dir,'merge.tif')
    os.makedirs(merge_dir, exist_ok=True)
    tifffile.imsave(merge_path, merge_image)

#----------------------------------
# MULTIPLE IMAGE FOLDERS
#----------------------------------
def merge(images_root, tile_size, force_z):
    '''
    Search the images_root and all sub directories for inference.yaml files.
    When an inference file is found, merge the tiles and save as the original
    image size.
    '''
    for root, dirs, files in os.walk(images_root):
        dirs.sort()
        # print(dirs)
        for d in tqdm(dirs):
            image_dir = os.path.join(root, d)
            inference_files = [f for f in os.listdir(image_dir) if 'inference' in f and f.endswith('.yaml')]
            if inference_files:
                inference_files.sort()
                inference_yaml = inference_files[-1]  # Most recent
                with open(os.path.join(image_dir,inference_yaml)) as inf:
                    info = yaml.safe_load(inf)
                original_image_path = info['image_path']
                original_shape = info['image_shape']
                merge_and_save(image_dir, original_shape, tile_size, force_z, original_image_path)

# EXAMPLE USE
# --------------
# images_root = '/n/core/micro/asa/auk/smc/20201106_3PO_IMARE-101128/Training/RICHARD/TEST/'  # '/n/core/micro/asa/fgm/smc/20190919_Screen/DeepLearn/Training/RICHARD/1612445035_mse_round_GOOD/'
# tile_size = (128,128,32)
# force_z = 32
# merge(images_root, tile_size, force_z)



# #----------------------------------
# # SINGLE IMAGE FOLDER
# #----------------------------------
# image_dir = '/n/core/micro/asa/fgm/smc/20190919_Screen/DeepLearn/Training/RICHARD/TEST/inference_images/Plate000_Well2_Object1' #'/n/core/micro/asa/fgm/smc/20190919_Screen/DeepLearn/Training/RICHARD/1612445035_mse_round_GOOD/0'
# assert os.path.exists(image_dir), 'Check image_dir'
# merge_and_save(image_dir)


# # # Delete bad attempts
# # images_root = '/n/core/micro/asa/fgm/smc/20190919_Screen/DeepLearn/Training/RICHARD/1612445035_mse_round_GOOD/'
# # for root, dirs, files in os.walk(images_root):
# #     dirs.sort()
# #     print(dirs)
# #     for d in tqdm(dirs):
# #         image_dir = os.path.join(root, d)
# #         merge_dir = os.path.join(image_dir,'merge')
# #         shutil.rmtree(merge_dir)
# 