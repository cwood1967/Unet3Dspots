from pims import ImageSequence
# from tensorflow import keras
from MightyMosaic import MightyMosaic
import matplotlib.pyplot as plt
import numpy as np
from read_roi import read_roi_zip
import random
import os

'''
Generate training data from disk.
- Open image
- Generate mask from ROI zip file
- Perform the same image augmentation on image and mask
- Split images into tiles
- Return tiles for training

Implement so that CPU operations do not block GPU training.
'''

def create_roi_image(roi_path, image_shape):
    rois = read_roi_zip(roi_path)
    spots = np.zeros(shape=image_shape)
    for i, roi in rois.items():
        x = roi['x'][0]
        y = roi['y'][0]
        spots[y-2:y+2, x-2:x+2] = 1.0
    return spots

def get_rand(val):
    if not val:
        return False
    return random.choice([True, False])

def open_image_mask(image_path):
    roi_path = '.'.join(image_path.split('.')[:-1]) + '.zip'
    images = ImageSequence(image_path)
    image = images[0]/np.max(images[0])  # Single image / Normalize
    roi_mask = create_roi_image(roi_path, image.shape)
    roi_mask = roi_mask/np.max(roi_mask)  # Normalize
    return image, roi_mask

def generate_tiles_masks_from_rois(data_path='/n/core/micro/mg2/cyc/rla/cnidocyte_analysis/train/', data_extension='.tif', horizontal_flip=True, vertical_flip=True, no_loop=False):
    while True:
        for root, dirs, files in os.walk(data_path):
            for f in files:
                # print(f)
                if f.endswith(data_extension):
                    # Load image and roi mask
                    image_path = os.path.join(root,f)
                    image, roi_mask = open_image_mask(image_path)
                    # Overwrite images to save memory
                    image = MightyMosaic.from_array(image, (512,512), overlap_factor=2)
                    roi_mask = MightyMosaic.from_array(roi_mask, (512,512), overlap_factor=2)
                    # # yield augmented image and roi_mask one tile at a time
                    # for i in range(image.shape[0]):
                    #     for j in range(image.shape[1]):
                    #         hflip = get_rand(horizontal_flip)
                    #         vflip = get_rand(vertical_flip)
                    #         if not hflip and not vflip:
                    #             #original
                    #             img = np.array([[image[i,j]]])
                    #             mask = np.array([[roi_mask[i,j]]])
                                 
                    #         elif hflip and not vflip:
                    #             img = np.array([[image[i,j][:,::-1]]])
                    #             mask = np.array([[roi_mask[i,j][:,::-1]]])
                    #         elif not hflip and vflip:
                    #             img = np.array([[image[i,j][::-1,:]]])
                    #             mask = np.array([[roi_mask[i,j][::-1,:]]])
                    #         elif hflip and vflip:
                    #             img = np.array([[image[i,j][::-1,::-1]]])
                    #             mask = np.array([[roi_mask[i,j][::-1,::-1]]])
                    #         yield img, mask
                    # yield all tiles at once
                    hflip = get_rand(horizontal_flip)
                    vflip = get_rand(vertical_flip)
                    if not hflip and not vflip:
                        #original
                        image = np.array(image)
                        image = np.reshape(image, (-1, image.shape[2], image.shape[3], 1))
                        roi_mask = np.array(roi_mask)
                        roi_mask = np.reshape(roi_mask, (-1, roi_mask.shape[2], roi_mask.shape[3], 1))
                            
                    # elif hflip and not vflip:
                    #     img = np.array([[image[i,j][:,::-1]]])
                    #     mask = np.array([[roi_mask[i,j][:,::-1]]])
                    # elif not hflip and vflip:
                    #     img = np.array([[image[i,j][::-1,:]]])
                    #     mask = np.array([[roi_mask[i,j][::-1,:]]])
                    # elif hflip and vflip:
                    #     img = np.array([[image[i,j][::-1,::-1]]])
                    #     mask = np.array([[roi_mask[i,j][::-1,::-1]]])
                    yield image, roi_mask
                        
            break  # No sub directories
        if no_loop:
            break

# def generate_augmented(data_path='/n/core/micro/mg2/cyc/rla/cnidocyte_analysis/train/', data_extension='.tif'):
#     # Create two image generator instances with the same arguments
#     data_gen_args = dict(
#         horizontal_flip = True,
#         vertical_flip = True,
#         # featurewise_center=True,
#         # featurewise_std_normalization=True,
#         # rotation_range=90,
#         # width_shift_range=0.1,
#         # height_shift_range=0.1,
#         )
#     image_datagen = ImageDataGenerator(**data_gen_args)
#     mask_datagen = ImageDataGenerator(**data_gen_args)
#     # Provide the same seed and keyword arguments to the fit and flow methods
#     seed = random.randint(1,1000)
#     for images, masks in generate_tiles_from_rois(data_path, data_extension):
#         image_datagen.fit(images, augment=True, seed=seed)
#         mask_datagen.fit(masks, augment=True, seed=seed)

for x in generate_tiles_masks_from_rois(horizontal_flip=False, vertical_flip=False, no_loop=True):
    print('Image shape', x[0].shape)
    print('ROI shape', x[1].shape)
    pass