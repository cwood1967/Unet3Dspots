import pims
from pims import ImageSequence, FramesSequenceND, ImageSequenceND, Bioformats
# from tensorflow import keras
from MightyMosaic import MightyMosaic
import matplotlib.pyplot as plt
import numpy as np
from read_roi import read_roi_zip
import random
import os
from utils import show_image, show_image_3d, overlay_image_3d_mask
from math import ceil
import tifffile
from tensorflow.keras.preprocessing.image import apply_affine_transform
from scipy.ndimage import gaussian_filter

'''
Generate training data from disk.
- Open image
- Generate mask from ROI zip file
- Perform the same image augmentation on image and mask
- Split images into tiles
- Return tiles for training

Implement so that CPU operations do not block GPU training.
'''

# # Test gaussian 
# tst = np.zeros(shape=(3,17,17))
# tst[1,8,8] = 1
# gauss = gaussian_filter(tst, sigma=1)

def get_gauss_spot():
    '''
    Create a masked gaussian to be used in image masks.
    '''
    spot_3d = np.array(
        [
            [
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 1, 1, 0, 0],
                [0, 1, 1, 1, 1, 1, 0],
                [0, 1, 1, 1, 1, 1, 0],
                [0, 1, 1, 1, 1, 1, 0],
                [0, 0, 1, 1, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
            ],
            [
                [0, 0, 1, 1, 1, 0, 0],
                [0, 1, 1, 1, 1, 1, 0],
                [1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 1, 1, 1],
                [0, 1, 1, 1, 1, 1, 0],
                [0, 0, 1, 1, 1, 0, 0],
            ],
            [
                [0, 0, 0, 0, 0, 0, 0],
                [0, 0, 1, 1, 1, 0, 0],
                [0, 1, 1, 1, 1, 1, 0],
                [0, 1, 1, 1, 1, 1, 0],
                [0, 1, 1, 1, 1, 1, 0],
                [0, 0, 1, 1, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 0],
            ]
        ],
        dtype=np.float32
    )
    return np.swapaxes(spot_3d,2,0)

def create_roi_image_3d(roi_path, image_shape):
    '''
    Create an image mask from an ImageJ ROI file where gaussian spots
    are placed at the center of each object.
    roi_path: Path to the imageJ ROI file
    image_shape: Tupple of X,Y,Z image shape
    return: Mask image containing gaussian spots
    '''
    rois = read_roi_zip(roi_path)
    spots = np.zeros(shape=image_shape)
    spots = spots.astype(np.float32)
    spot_3d = get_gauss_spot()
    spot_shape = spot_3d.shape
    for i, roi in rois.items():
        x = roi['x'][0]
        y = roi['y'][0]
        position = roi['position']
        # roi.position can be int or dict
        if isinstance(position, int):
            z = position - 1  # One indexed
        else:
            z = position['slice'] - 1  # One indexed
        # Original square spots
        # spots[y-2:y+3, x-2:x+3, z-1:z+2] = 1.0
        # spots[y-3:y+4, x-3:x+4, z] = 1.0  # Bigger middle slice
        # New round spot
        try:
            # No partial spots allowed
            spots[y-3:y+4, x-3:x+4, z-1:z+2] = spot_3d.copy()
        except:
            pass
    return spots

def get_rand(val):
    '''
    Generate a random True/False value
    return: True or False at random
    '''
    if not val:
        return False
    return random.choice([True, False])

def compress_z(image, n=10):
    '''
    Given a set of batched images, compress n z-slices into 1
    by max projecting.
    image: The image to compress
    n: The number of z-slices to max project
    '''
    # Expected shape: y, x, z
    z = image.shape[2]
    new_slices = int(z/n)
    new_shape = [*image.shape]
    new_shape[2] = new_slices
    # new_single = new_shape
    # new_single[2] = 1
    image_compressed = np.zeros(new_shape)
    for s in range(new_slices):
        # img = np.max(image, axis=2)
        # image_compressed[:,:,s] = np.reshape(img, newshape=new_single)
        image_compressed[:,:,s] = np.max(image, axis=2)
    return image_compressed

# def open_image_mask_3d(image_path, channel=1):
#     roi_path = '.'.join(image_path.split('.')[:-1]) + '.zip'
#     images = pims.open(image_path)
#     with pims.open(image_path) as frames:
#         frames.bundle_axes = 'czyx'
#         # print('PIMS shape:', frames.frame_shape)
#         for image in frames:
#             shp = image.shape
#             # Mosaic expect first two axes to be y, x
#             # Can currently only accept either channel or z
#             one_chan = image[channel]
#             reshape_img = np.moveaxis(one_chan, 0, -1) # Select one channel and all z
#             # # print('Image shape:', reshape_img.shape)
#             # mosaic = MightyMosaic.from_array(reshape_img, (512,512), overlap_factor=2)
#             # # print('Mosaic shape:', mosaic.shape)
#     image = reshape_img/np.max(reshape_img)  # Single image / Normalize
#     # image = compress_z(image)
#     roi_mask = create_roi_image_3d(roi_path, image.shape)
#     roi_mask = roi_mask/np.max(roi_mask)  # Normalize
#     return image, roi_mask

def open_tiffimage_mask_3d(image_path, channel=0, force_z=None, z_compress=None):
    '''
    Open the 3D image and its 3D mask.
    image_path: Path to image
    channel: Which single channel to process. Zero indexed: 0,1,2, ...
    force_z: Forces the image to the specified number of z-slices.
        If force_z is None, no action is taken.
    z_compress: Number of z-slices to compress, reducing computation time
    return: Tupple of 3D image and 3D mask
    '''
    roi_path = '.'.join(image_path.split('.')[:-1]) + '.zip'
    with pims.Bioformats(image_path) as frames:
        num_dim = len(frames.axes)
        if num_dim == 3:
            frames.bundle_axes = 'yxz'
        else :
            frames.bundle_axes = 'cyxz'
        # print('PIMS shape:', frames.frame_shape)
        for img in frames:  # Only a single image if no time component
            if num_dim > 3:
                img = img[channel]
            #     one_chan = img[channel]
            #     reshape_img = np.moveaxis(one_chan, 0, -1) # Select one channel and all z
            # else:
            #     reshape_img = np.moveaxis(img, 0, -1)
            break

    image = img/np.max(img)  # Single image / Normalize
    image = img/np.max(img)  # Single image / Normalize
    if force_z:
        if force_z < image.shape[2]:
            # Take the first z slices
            image = image[:force_z]
        elif force_z > image.shape[2]:
            # Add z-slices to the end of the stack
            z_pad = force_z - image.shape[2]
            image = np.pad(image, ((0,0),(0,0),(0,z_pad)), 'constant')
    if z_compress:
            image = compress_z(image, n=z_compress)
    # Normalize differently
    # image = ((reshape_img - np.mean(reshape_img)) / np.std(reshape_img)) + 0.5
    # # Norm with preset values
    # mean=153.0
    # std=100.0
    # image = (reshape_img - mean) / (std*1) + 0.5
    # print('opened Image:')
    # show_image_3d(image)
    image = image.astype(np.float32)
    if os.path.isfile(roi_path):
        roi_mask = create_roi_image_3d(roi_path, image.shape)
        roi_mask = roi_mask/np.max(roi_mask)  # Normalize
    else:
        print('''WARNING: ROI FILE DOES NOT EXIST... Hopefully you're running inference''')
        roi_mask = None
    return image, roi_mask

# def generate_tiles_masks_single(image_path, data_extension='-3d.tif', tile_shape=(512,512), horizontal_flip=True, vertical_flip=True):
#     # print(f)
#     # Load image and roi mask
#     image, roi_mask = open_tiffimage_mask_3d(image_path)
#     # Overwrite images to save memory
#     image = MightyMosaic.from_array(image, tile_shape, overlap_factor=2) # tile_shape=(512,512, image.shape[2]),
#     roi_mask = MightyMosaic.from_array(roi_mask, tile_shape, overlap_factor=2)
#     # yield augmented image and roi_mask one tile at a time
#     for i in range(image.shape[0]):
#         for j in range(image.shape[1]):
#             hflip = get_rand(horizontal_flip)
#             vflip = get_rand(vertical_flip)
#             if not hflip and not vflip:
#                 #original
#                 img = np.array(image[i,j])
#                 mask = np.array(roi_mask[i,j])
#             elif hflip and not vflip:
#                 img = np.array(image[i,j][:,::-1, :])
#                 mask = np.array(roi_mask[i,j][:,::-1])
#             elif not hflip and vflip:
#                 img = np.array(image[i,j][::-1,:, :])
#                 mask = np.array(roi_mask[i,j][::-1,:])
#             elif hflip and vflip:
#                 img = np.array(image[i,j][::-1,::-1, :])
#                 mask = np.array(roi_mask[i,j][::-1,::-1])

#             img = np.reshape(img, (-1, img.shape[0], img.shape[1], img.shape[2], 1)) # batch, y, x, z, c
#             mask = np.reshape(mask, (-1, mask.shape[0], mask.shape[1], 1, 1))
            
#             yield img, mask
#     # # yield all tiles at once
#     # hflip = get_rand(horizontal_flip)
#     # vflip = get_rand(vertical_flip)
#     # if not hflip and not vflip:
#     #     #original
#     #     image = np.array(image)
#     #     image = np.reshape(image, (-1, image.shape[2], image.shape[3], image.shape[4], 1)) # batch, y, x, z, c
#     #     roi_mask = np.array(roi_mask)
#     #     roi_mask = np.reshape(roi_mask, (-1, roi_mask.shape[2], roi_mask.shape[3], 1, 1))
            
#     # elif hflip and not vflip:
#     #     img = np.array([[image[i,j][:,::-1]]])
#     #     mask = np.array([[roi_mask[i,j][:,::-1]]])
#     # elif not hflip and vflip:
#     #     img = np.array([[image[i,j][::-1,:]]])
#     #     mask = np.array([[roi_mask[i,j][::-1,:]]])
#     # elif hflip and vflip:
#     #     img = np.array([[image[i,j][::-1,::-1]]])
#     #     mask = np.array([[roi_mask[i,j][::-1,::-1]]])
#     # yield image, roi_mask

# def generate_tiles_masks_from_rois(data_path='/n/core/micro/mg2/cyc/rla/cnidocyte_analysis/validate/', data_extension='-3d.tif', tile_shape=(512,512), horizontal_flip=True, vertical_flip=True, no_loop=False):
#     while True:
#         for root, dirs, files in os.walk(data_path):
#             for f in files:
#                 if f.endswith(data_extension):
#                     image_path = os.path.join(root,f)
#                     yield from generate_tiles_masks_single(image_path, data_extension, tile_shape, horizontal_flip, vertical_flip)
                        
#             break  # No sub directories
#         if no_loop:
#             break

# for root, dirs, files in os.walk('/n/core/micro/asa/fgm/smc/20190919_Screen/DeepLearn/Training/RICHARD'):
#     print(root)
#     print(files)

def rotate_image(image, theta, row_axis, col_axis, channel_axis, fill_mode='constant'):
    """Performs a random rotation of a Numpy image tensor.
    # Arguments
        x: Input tensor. Must be 3D.
        rg: Rotation range, in degrees.
        row_axis: Index of axis for rows in the input tensor.
        col_axis: Index of axis for columns in the input tensor.
        channel_axis: Index of axis for channels in the input tensor.
        fill_mode: Points outside the boundaries of the input
            are filled according to the given mode
            (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
        cval: Value used for points outside the boundaries
            of the input if `mode='constant'`.
        interpolation_order: int, order of spline interpolation.
            see `ndimage.interpolation.affine_transform`
    # Returns
        Rotated Numpy image tensor.
    """
    image = apply_affine_transform(image,
                                theta=theta,
                                row_axis=row_axis,
                                col_axis=col_axis,
                                channel_axis=channel_axis,
                                fill_mode=fill_mode,
                                cval=0.0,
                                order=1)
    return image

def shift_image(image, tx, ty, row_axis, col_axis, channel_axis, fill_mode='constant'):
    """Performs a random spatial shift of a Numpy image tensor.
    # Arguments
        x: Input tensor. Must be 3D.
        tx: number of pixels to shift in x
        ty: number of pixels to shift in y
        wrg: Width shift range, as a float fraction of the width.
        hrg: Height shift range, as a float fraction of the height.
        row_axis: Index of axis for rows in the input tensor.
        col_axis: Index of axis for columns in the input tensor.
        channel_axis: Index of axis for channels in the input tensor.
        fill_mode: Points outside the boundaries of the input
            are filled according to the given mode
            (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
        cval: Value used for points outside the boundaries
            of the input if `mode='constant'`.
        interpolation_order: int, order of spline interpolation.
            see `ndimage.interpolation.affine_transform`
    # Returns
        Shifted Numpy image tensor.
    """
    
    image = apply_affine_transform(image,
                               tx=tx,
                               ty=ty,
                               row_axis=row_axis,
                               col_axis=col_axis,
                               channel_axis=channel_axis,
                               fill_mode=fill_mode,
                               cval=0.0,
                               order=1)
    return image

def get_overlap(tile_size, overlap_fraction):
    '''
    Get the tile overlap for each dimension in tile_size.
    Funcion assumes that all dimensions are in the same order.
    tile_size: The size of each tile dimension
    overlap_fraction: A number between 0 and 1 representing the
        fraction of tile_size which should overlap in each dimension
    return: List of overlap length in each dimension
    '''
    overlap = [ceil(t * o) for t, o in zip(tile_size, overlap_fraction)]
    return overlap

def get_padded_shape(image_shape, overlap):
    '''
    This function can accept any shape of image, but 
    x and y must be in the first two dimensions. Only
    x and y will be padded.
    '''
    pad_shape = []
    for s, o in zip(image_shape, overlap):
        if s < 2:
            pad_shape.append((0,s%o))
        else:
            pad_shape.append((0,0))
    return pad_shape

def tile_position_generator(image_shape, tile_size, overlap):
    for z in range(0,image_shape[2]-tile_size[2]+2,tile_size[2]-overlap[2]):
        for x in range(0,image_shape[1]-tile_size[1]+1,tile_size[1]-overlap[1]):
            for y in range(0,image_shape[0]-tile_size[0]+1,tile_size[0]-overlap[0]):
                x1 = x+tile_size[1]
                y1 = y+tile_size[0]
                z1 = z+tile_size[2]
                yield x, x1, y, y1, z, z1

def generate_tiles(image, mask_image=None, tile_size=(128,128,8), overlap_fraction=(0.5,0.5,0.5), horizontal_flip=True, vertical_flip=True, random_rotation=True, random_shift=True):
    '''
    Generate overlapping tiles from an n-dimensional image
    image: The numpy image to tile.
    mask_image: The numpy mask image.
    tile_size: The size of each tile where each tile dimension is smaller than the image
    overlap_fraction: Tuple or list where each element represents the fraction that each 
        tile dimensionthat should overlap with the next/previous tile. Each element
        should be be between 0 and 1, where zero is no overlap.
        Safe possible values for power-of-2 dimensions are 0, 0.25, and 0.5
    horizontal_flip: If true, randomly flip the image horizontallye. If false, always return original.
    vertical_flip: If true, randomly flip the image vertically. If false, always return original.
    random_rotation: Apply random rotation to image and mask
    random_shift: Apply random shift to image and mask
    '''
    if mask_image is not None:
        assert image.shape == mask_image.shape, f'Image ({image.shape}) and mask ({image.shape}) should have the same shape.'
    assert len(image.shape)==len(tile_size), f'Image has dimension {len(image.shape)}, but tile_size has dimension {len(tile_size)}'
    assert len(image.shape)==len(overlap_fraction), f'Image has dimension {len(image.shape)}, but overlap_fraction has dimension {len(overlap_fraction)}'
    assert min(overlap_fraction)>=0 and max(overlap_fraction)<1, f'overlap_fraction must be between 0 and 1'
    for i, t in zip(image.shape, tile_size):
        assert i>=t, f'Image shape ({i}) must be greater than or equal to tile size ({t})' 
    # print('image_shape:', image.shape)
    # print('tile_size:', tile_size)
    row_axis = 0
    col_axis = 1
    channel_axis = 2
    overlap = get_overlap(tile_size, overlap_fraction)

    pad_shape = get_padded_shape(image.shape, overlap)
    image = np.pad(image, pad_shape, 'constant')
    image_y, image_x, image_z = image.shape
    
    # print('overlap:', overlap)
    # loop through each dimension, creating overlapping tiles
    # for i in range(0,image.size):

    if random_rotation:
        # Apply random rotation to image and mask
        theta = np.random.uniform(-90, 90)  # 90 degrees
        image = rotate_image(image, theta, row_axis, col_axis, channel_axis) # fill_mode='nearest'
        if mask_image is not None:
            mask_image = rotate_image(mask_image, theta, row_axis, col_axis, channel_axis)
    if random_shift:
        # Apply random shift to image and mask
        hrg = wrg = 0.1  # Fraction of height/width to shift the image
        tx = np.random.uniform(-hrg, hrg) * tile_size[0]
        ty = np.random.uniform(-wrg, wrg) * tile_size[1]
        image = shift_image(image, tx, ty, row_axis, col_axis, channel_axis) # fill_mode='nearest'
        if mask_image is not None:
            mask_image = shift_image(mask_image, tx, ty, row_axis, col_axis, channel_axis)
    
    count = 0
    for x, x1, y, y1, z, z1 in tile_position_generator(image.shape, tile_size, overlap):
        count +=1 
        # print(f'({y}:{y+tile_size[0]}, {x}:{x+tile_size[1]}, {z}:{z+tile_size[2]})')
        img = image[y:y1, x:x1, z:z1]
        if mask_image is not None:
            mask = mask_image[y:y1, x:x1, z:z1]
        hflip = get_rand(horizontal_flip)
        vflip = get_rand(vertical_flip)
        # Each window can be flipped differently
        if not hflip and not vflip:
            #original
            img = np.array(img)
            if mask_image is not None:
                mask = np.array(mask)
        elif hflip and not vflip:
            img = np.array(img[:,::-1, :])
            if mask_image is not None:
                mask = np.array(mask[:,::-1, :])
        elif not hflip and vflip:
            img = np.array(img[::-1,:, :])
            if mask_image is not None:
                mask = np.array(mask[::-1,:, :])
        elif hflip and vflip:
            img = np.array(img[::-1,::-1, :])
            if mask_image is not None:
                mask = np.array(mask[::-1,::-1, :])
        
        img = np.reshape(img, (-1, img.shape[0], img.shape[1], img.shape[2], 1)) # batch, y, x, z, c
        if mask_image is not None:
            mask = np.reshape(mask, (-1, mask.shape[0], mask.shape[1], mask.shape[2], 1))
        else:
            mask = None
        yield img, mask

    # for z in range(0,image.shape[2]-tile_size[2]+2,tile_size[2]-overlap[2]):
    #     for x in range(0,image.shape[1]-tile_size[1]+1,tile_size[1]-overlap[1]):
    #         for y in range(0,image.shape[0]-tile_size[0]+1,tile_size[0]-overlap[0]):
                # count +=1 
                # # print(f'({y}:{y+tile_size[0]}, {x}:{x+tile_size[1]}, {z}:{z+tile_size[2]})')
                # img = image[y:y+tile_size[0], x:x+tile_size[1], z:z+tile_size[2]]
                # if mask_image is not None:
                #     mask = mask_image[y:y+tile_size[0], x:x+tile_size[1], z:z+tile_size[2]]
                # hflip = get_rand(horizontal_flip)
                # vflip = get_rand(vertical_flip)
                # # Each window can be flipped differently
                # if not hflip and not vflip:
                #     #original
                #     img = np.array(img)
                #     if mask_image is not None:
                #         mask = np.array(mask)
                # elif hflip and not vflip:
                #     img = np.array(img[:,::-1, :])
                #     if mask_image is not None:
                #         mask = np.array(mask[:,::-1, :])
                # elif not hflip and vflip:
                #     img = np.array(img[::-1,:, :])
                #     if mask_image is not None:
                #         mask = np.array(mask[::-1,:, :])
                # elif hflip and vflip:
                #     img = np.array(img[::-1,::-1, :])
                #     if mask_image is not None:
                #         mask = np.array(mask[::-1,::-1, :])
                
                # img = np.reshape(img, (-1, img.shape[0], img.shape[1], img.shape[2], 1)) # batch, y, x, z, c
                # if mask_image is not None:
                #     mask = np.reshape(mask, (-1, mask.shape[0], mask.shape[1], mask.shape[2], 1))
                # else:
                #     mask = None
                # yield img, mask

def merge_tiles(image_dir, merge_shape, image_extension='.tif', tile_size=(128,128,8), overlap_fraction=(0.5,0.5,0.5), mask_edge_fraction=(0.2, 0.2, 0.2)):
    '''
    Generate overlapping tiles from an n-dimensional image
    image_dir: The directory containing all tiff images to merge. Images contain tiles in the first dimension.
    merge_shape: The shape of the original image tiles were created from. Assumes tile_size
        is an even multiple of merge_shape.
    image_extension: The extension of the image file (ex .tif)
    overlap_fraction: Tuple or list where each element represents the fraction that each 
        tile dimension should overlap with the next/previous tile. Each element should
        be be between 0 and 1, where zero is no overlap.
        Safe possible values for power-of-2 dimensions are 0, 0.25, and 0.5
    mask_edge_fraction: Tuple or list where each element represents the fraction that each 
        tile dimension that should overlap with the next/previous tile. Each element
        should be be between 0 and 1, where zero is no masking along the edge.   
    '''
    def check_assertions(image):
        assert len(image.shape[1:]) == len(merge_shape), f'Image ({image.shape}) and merge ({merge_shape}) should have compatible shape.'
        assert len(image.shape[1:])==len(tile_size), f'Image has dimension {len(image.shape)}, but tile_size has dimension {len(tile_size)}'
        assert len(image.shape[1:])==len(overlap_fraction), f'Image has dimension {len(image.shape)}, but overlap_fraction has dimension {len(overlap_fraction)}'
        assert min(overlap_fraction)>=0 and max(overlap_fraction)<1, f'overlap_fraction must be between 0 and 1'
        for i, t in zip(image.shape[1:], tile_size):
            assert i>=t, f'Image shape ({i}) must be greater than or equal to tile size ({t})' 
        for m, t in zip(merge_shape, tile_size):
            assert m>=t, f'Merge shape ({m}) must be greater than or equal to tile size ({t})' 
        for m, o in zip(mask_edge_fraction, overlap_fraction):
            assert o >= m*2, f'Overlap fraction ({o}) must be at least twice as big as mask_edge_fraction ({m})' 

    def image_generator_tiff(image_dir, image_extension):
        for root, dirs, files in os.walk(image_dir):
            files.sort()
            image_files = [f for f in files if f.endswith(image_extension)]
            for f in image_files:
                image_path = os.path.join(image_dir, f)
                image = tifffile.imread(image_path)
                image = image[:,:,0,:,:]
                image = np.moveaxis(image, 3, 1)
                image = np.moveaxis(image, 3, 1)
                yield image
            break # No subdirectories
    
    overlap = get_overlap(tile_size, overlap_fraction)
    pad_shape = get_padded_shape(merge_shape, overlap)
    merge_image = np.zeros(shape=merge_shape, dtype=np.float32)
    merge_image = np.pad(merge_image, pad_shape, 'constant')
    tile_mask = np.zeros(shape=tile_size, dtype=np.float32)
    mx, my, mz = tile_size
    dx = ceil(mx*mask_edge_fraction[0])
    dy = ceil(my*mask_edge_fraction[1])
    dz = ceil(mz*mask_edge_fraction[2])
    tile_mask[0+dx:mx-dx, 0+dy:my-dy, 0+dz:mz-dz] = 1.0
    
    # print('overlap:', overlap)
    # loop through each dimension, creating overlapping tiles
    # for i in range(0,image.size):    
    def lay_tiles(image_gen, apply_mask=True):
        '''
        Loop through each merge_shape dimension, laying overlapping tiles.
        This is done in two stages. First a mask is applied to all tiles,
        removing all pixels along the edges. Those masked tiles are placed
        in the merge_image. Second, the original images are used to fill
        in only remaining pixels that are zero from the first step.
        image_gen: Image generator
        apply_mask: If true mask tile edges and use the max pixel.
            If false, use original tile image and only place new pixels 
            where the merge_image is zero.
        return: merge_image
        '''
        tile_count = 0
        image = next(image_gen)
        check_assertions(image)
        
        for x, x1, y, y1, z, z1 in tile_position_generator(merge_image.shape, tile_size, overlap):
            # print(f'(y {y}:{y+tile_size[0]}, x {x}:{x+tile_size[1]}, z {z}:{z+tile_size[2]})')
            if apply_mask:
                masked_image = image[tile_count] * tile_mask
                merge_image[y:y1, x:x1, z:z1] = np.maximum(masked_image, merge_image[y:y1, x:x1, z:z1])
            else:
                masked_image = image[tile_count]
                merge_image[y:y1, x:x1, z:z1] = np.where(merge_image[y:y1, x:x1, z:z1]==0, masked_image, merge_image[y:y1, x:x1, z:z1])
            tile_count +=1
            if tile_count >= len(image):
                # When we complete one image, reset tile count and load next image
                tile_count = 0
                image = next(image_gen)

        # for z in range(0,merge_shape[2]-tile_size[2]+2,tile_size[2]-overlap[2]):
        #     for x in range(0,merge_shape[1]-tile_size[1]+1,tile_size[1]-overlap[1]):
        #         for y in range(0,merge_shape[0]-tile_size[0]+1,tile_size[0]-overlap[0]):
                    
        #             # print(f'(y {y}:{y+tile_size[0]}, x {x}:{x+tile_size[1]}, z {z}:{z+tile_size[2]})')
        #             if apply_mask:
        #                 masked_image = image[tile_count] * tile_mask
        #                 merge_image[y:y+tile_size[0], x:x+tile_size[1], z:z+tile_size[2]] = np.maximum(masked_image, merge_image[y:y+tile_size[0], x:x+tile_size[1], z:z+tile_size[2]])
        #             else:
        #                 masked_image = image[tile_count]
        #                 merge_image[y:y+tile_size[0], x:x+tile_size[1], z:z+tile_size[2]] = np.where(merge_image[y:y+tile_size[0], x:x+tile_size[1], z:z+tile_size[2]]==0, masked_image, merge_image[y:y+tile_size[0], x:x+tile_size[1], z:z+tile_size[2]])
        #             tile_count +=1
        #             if tile_count >= len(image):
        #                 # When we complete one image, reset tile count and load next image
        #                 tile_count = 0
        #                 image = next(image_gen)

    # Lay all tiles, applying the mask to each tile
    image_gen = image_generator_tiff(image_dir, image_extension)
    lay_tiles(image_gen, apply_mask=True)
    # Fill in zeroes on the edges
    image_gen = image_generator_tiff(image_dir, image_extension)
    lay_tiles(image_gen, apply_mask=False)

    # Shrink image if padded
    merge_image = merge_image[:merge_shape[0], :merge_shape[1], :merge_shape[2]]

    merge_image = np.moveaxis(merge_image, 0,-1)
    merge_image = np.moveaxis(merge_image, 0,-1)

    return merge_image

def generate_image_mask_tiles(data_path='/n/core/micro/asa/fgm/smc/20190919_Screen/DeepLearn/Training/RICHARD/Train/', data_extension='.tif', 
                                batch_size=32, tile_size=(128,128,8), overlap_fraction=(0.25,0.25,0.5), no_loop=False, 
                                horizontal_flip=True, vertical_flip=True, random_rotation=True, random_shift=True, 
                                yield_single_file=False, yield_image_name=False, images_in_memory=True,
                                channel=None, force_z=None):
    '''
    Helper function to generate matching image and mask tiles from the
    generate_tiles generator.
    image: The image to tile
    mask: The mask to tile
    tile_size: (See generate_tiles())
    overlap_fraction: (See generate_tiles())
    no_loop: If true, stop after a single batch. If false, continue generating images indefinitely.
    horizontal_flip: If true, randomly flip the image horizontallye. If false, always return original.
    vertical_flip: If true, randomly flip the image vertically. If false, always return original.
    random_rotation: Apply random rotation to image and mask
    random_shift: Apply random shift to image and mask
    yield_single_file: Yield batches containing only a single image file if True.
        If false, batches can contain parts of multiple images. This typically should be left as False
        for training the model, but yielding a single image is useful in other cases.
    yield_image_name: Yield the image name along with tiles if True. If False, do not yield the image 
        name with tiles. Set to False for training, but True for inference.
    images_in_memory: If True, load all images into memory before starting training to save IO time.
        If false, load each image as it is needed. Set to True if at all possible, which is if you have 
        enough RAM to hold all training images and masks.
    channel: Integer representing the single channel to keep for training. All other channels are discarded.
    force_z: Integer representing the number of z-slices the training data should have. This number should 
        be an equal subdivision of 2 of (or equal to) the number of z-slices in the neural network.
        Example: Network has 8 z-slices. force_z can be 8 or 12 or 16 or 32 assuming that the overlap
        fraction in z is 0.5.
    return: Tuple of (image_tile, mask_tile)
    '''

    # ##########################
    # # Image Augmentation Start
    # ##########################

    # data_gen_args = dict(
    #     featurewise_center=True,
    #     featurewise_std_normalization=True,
    #     rotation_range=90,
    #     width_shift_range=0.1,
    #     height_shift_range=0.1,
    # )

    # image_aug_gen = ImageDataGenerator(**data_gen_args)
    # mask_aug_gen = ImageDataGenerator(**data_gen_args)
    # # Provide the same seed and keyword arguments to the fit and flow methods
    # seed = 1
    # image_aug_gen.fit(images, augment=True, seed=seed)
    # mask_aug_gen.fit(masks, augment=True, seed=seed)
    # image_augmentor = image_aug_gen.flow_from_directory(
    #     'data/images',
    #     class_mode=None,
    #     seed=seed)
    # mask_augmentor = mask_aug_gen.flow_from_directory(
    #     'data/masks',
    #     class_mode=None,
    #     seed=seed)
    # ##########################
    # # Image Augmentation End
    # ##########################

    def yield_batch(image_batch, mask_batch, image_count, image_name, image_shape, yield_image_name):
        '''
        When image and mask batches are generated, only (image, mask) tuples can be given
        to Keras durring training. But for inference we need other information to be passed 
        along with the batch. This function yields the appropriate information based on 
        whether yield_image_name is True or False.
        image_batch: Batch of images
        mask_batch: Batch of masks in the same order as image_batch
        image_count: Only used if yield_image_name = True
        image_name: Name of image. Only used if yield_image_name = True
        image_shape: Shape of image. Only used if yield_image_name = True
        yield_image_name: If True, yield all info. If false only yield (image, mask) tuple
        return: Tuple of (image, batch) or (image, batch, info...)
        '''
        if yield_image_name:
            num = str(image_count).zfill(4)
            splt = image_name.split('.')
            name = '.'.join(splt[:-1])
            # ext = splt[-1]
            # image_batch_name = f'{name}_{num}'
            yield image_batch, mask_batch, name, num, image_shape
        else:
            yield image_batch, mask_batch

    def load_all_image_mask(data_path, data_extension):
        '''
        Load all images into memory to save read time
        data_path: path to the image data
        data_extension: The image extension to be loaded
        return: Tuple of (image_count, image_mask_tuples)
        '''
        image_mask_tuples = []
        for root, dirs, files in os.walk(data_path):
            files.sort()
            for f in files:
                if f.endswith(data_extension):
                    image_path = os.path.join(root,f)
                    image, mask = open_tiffimage_mask_3d(image_path, channel=channel, force_z=force_z, z_compress=None)
                    image_mask_tuples.append((image,mask,f))
            break  # No sub directories
        image_count = len(image_mask_tuples)
        return image_count, image_mask_tuples

    def image_mask_generator(image_mask_tuples, no_loop):
        '''
        Generate image mask tuples stored in memory
        return: Tuple of (count, image, mask, image_name)
        '''
        while True:
            for i, image_mask in enumerate(image_mask_tuples):
                # count, image, mask, name
                yield i, image_mask[0], image_mask[1], image_mask[2]
            if no_loop:
                break
    
    def get_file_count(data_path, data_extension):
        '''
        Get the number of images with a specific extension in a directory
        data_path: The path to the image directory
        data_extension: The image extension (ex .tif)
        return: image count
        '''
        for root, dirs, files in os.walk(data_path):
            files.sort()
            files = [f for f in files if f.endswith(data_extension)]
            image_count = len(files)
            return(image_count) # No loop

    def generate_single_image_mask(data_path, data_extension, no_loop, channel, force_z):
        '''
        Generate image mask tuples from disk and only yeild batches that contain data
        from a single image. This mode is used for inference.
        return: Tuple of (count, image, mask, image_name)
        '''
        for root, dirs, files in os.walk(data_path):
            files.sort()
            files = [f for f in files if f.endswith(data_extension)]
            while True:
                for i, f in enumerate(files):
                    image_path = os.path.join(root,f)
                    image, mask = open_tiffimage_mask_3d(image_path, channel=channel, force_z=force_z, z_compress=None)
                    yield i, image, mask, f
                if no_loop:
                    break
            break
                

    image_mask_tuples = None
    image_batch = None
    mask_batch = None
    count = 0
    if images_in_memory and not yield_single_file:
        if not image_mask_tuples:
            image_count, image_mask_tuples = load_all_image_mask(data_path, data_extension)
        im_generator = image_mask_generator(image_mask_tuples, no_loop=no_loop)
    else:
        image_count = get_file_count(data_path, data_extension)
        im_generator = generate_single_image_mask(data_path, data_extension, channel=channel, force_z=force_z, no_loop=no_loop)
    while True:
        batch_count = 1
        for i, image, mask, image_name in im_generator:
            # yield from generate_tiles_masks_single(image_path, data_extension, tile_shape, horizontal_flip, vertical_flip)
            for image_tile, mask_tile in generate_tiles(image, mask, tile_size, overlap_fraction, horizontal_flip, vertical_flip, random_rotation, random_shift):
                if image_batch is None:
                    # Initialize numpy arrays
                    image_batch = np.zeros(shape=[batch_size, *image_tile.shape[1:]], dtype=np.float32)
                    if mask_tile is not None:
                        mask_batch = np.zeros(shape=[batch_size, *image_tile.shape[1:]], dtype=np.float32)
                image_batch[count] = image_tile[0]
                if mask_tile is not None:
                    mask_batch[count] = mask_tile[0]
                count+=1
                if count == batch_size:
                    yield from yield_batch(image_batch, mask_batch, batch_count, image_name, image.shape, yield_image_name)
                    # Reset batches
                    image_batch = None
                    mask_batch = None
                    count = 0
                    batch_count+=1
            if yield_single_file:
                if image_batch is not None:
                    yield from yield_batch(image_batch, mask_batch, batch_count, image_name, image.shape, yield_image_name)
                    # Reset batches
                    image_batch = None
                    mask_batch = None
                    count = 0
                    batch_count = 1
        if no_loop:
            if image_batch is not None:
                # Yield the final batch that may not equal the batch size
                yield from yield_batch(image_batch, mask_batch, batch_count, image_name, image.shape, yield_image_name)
                batch_count = 1
            break
    
# # Test single image
# path = '/n/core/micro/asa/fgm/smc/20190919_Screen/DeepLearn/Training/RICHARD/Train/A.tif'
# image, mask = open_tiffimage_mask_3d(path, z_compress=None)
# for x in generate_tiles(image):
#     print(x.shape)

# Test folder of images / masks
# for x in generate_image_mask_tiles(no_loop=True):
#     pass
# print('done')

# # Test validation output
# image_dir = '/n/core/micro/asa/fgm/smc/20190919_Screen/DeepLearn/Training/RICHARD/Validate/'
# image_extension = '.tif'
# tile_size = (128,128,8)
# overlap_fraction=(0.5,0.5,0.5)
# for image_tile, mask_tile, image_tile_name in generate_image_mask_tiles(image_dir, '.tif', batch_size=32,
#                                             tile_size=tile_size, overlap_fraction=overlap_fraction, 
#                                             no_loop=True, horizontal_flip=False, vertical_flip=False, 
#                                             random_rotation=False, random_shift=False,
#                                             yield_single_file=True, yield_image_name=True):
#     save_dir = os.path.join(image_dir,'input_test_norm_128')
#     os.makedirs(save_dir,exist_ok=True)
#     # For batch of images 
#     # image_tile = image_tile[0,:,:,:,:]
#     image_tile = np.moveaxis(image_tile,1,-1)
#     image_tile = np.moveaxis(image_tile,1,-1)
#     save_path = os.path.join(save_dir, f'{image_tile_name}.tif')
#     tifffile.imsave(save_path, image_tile)
#     save_path = os.path.join(save_dir, f'{image_tile_name}_mask.tif')
#     mask_tile = mask_tile.astype(np.float32)
#     # mask_tile = mask_tile[0,:,:,:,:]
#     mask_tile = np.moveaxis(mask_tile,1,-1)
#     mask_tile = np.moveaxis(mask_tile,1,-1)
#     tifffile.imsave(save_path, mask_tile)
#     # # For single image
#     # image_tile = image_tile[0,:,:,:,:]
#     # image_tile = np.moveaxis(image_tile,0,-1)
#     # image_tile = np.moveaxis(image_tile,0,-1)
#     # save_path = os.path.join(save_dir, f'{image_tile_name}.tif')
#     # tifffile.imsave(save_path, image_tile)
#     # save_path = os.path.join(save_dir, f'{image_tile_name}_mask.tif')
#     # mask_tile = mask_tile.astype(np.float32)
#     # mask_tile = mask_tile[0,:,:,:,:]
#     # mask_tile = np.moveaxis(mask_tile,0,-1)
#     # mask_tile = np.moveaxis(mask_tile,0,-1)
#     # tifffile.imsave(save_path, mask_tile)


# # Test augmented trainging  input
# image_dir = '/n/core/micro/asa/fgm/smc/20190919_Screen/DeepLearn/Training/RICHARD/Train/'
# image_extension = '.tif'
# tile_size = (128,128,8)
# overlap_fraction=(0.5,0.5,0.5)
# for image_tile, mask_tile, image_tile_name in generate_image_mask_tiles(image_dir, '.tif', batch_size=32,
#                                             tile_size=tile_size, overlap_fraction=overlap_fraction, 
#                                             no_loop=True, horizontal_flip=False, vertical_flip=False, 
#                                             random_rotation=True, random_shift=True,
#                                             yield_single_file=True, yield_image_name=True):
#     save_dir = os.path.join(image_dir,'input_test_augment_128_new')
#     os.makedirs(save_dir,exist_ok=True)
#     # For batch of images 
#     # image_tile = image_tile[0,:,:,:,:]
#     image_tile = np.moveaxis(image_tile,1,-1)
#     image_tile = np.moveaxis(image_tile,1,-1)
#     save_path = os.path.join(save_dir, f'{image_tile_name}.tif')
#     tifffile.imsave(save_path, image_tile)
#     save_path = os.path.join(save_dir, f'{image_tile_name}_mask.tif')
#     mask_tile = mask_tile.astype(np.float32)
#     # mask_tile = mask_tile[0,:,:,:,:]
#     mask_tile = np.moveaxis(mask_tile,1,-1)
#     mask_tile = np.moveaxis(mask_tile,1,-1)
#     tifffile.imsave(save_path, mask_tile)
#     # # For single image
#     # image_tile = image_tile[0,:,:,:,:]
#     # image_tile = np.moveaxis(image_tile,0,-1)
#     # image_tile = np.moveaxis(image_tile,0,-1)
#     # save_path = os.path.join(save_dir, f'{image_tile_name}.tif')
#     # tifffile.imsave(save_path, image_tile)
#     # save_path = os.path.join(save_dir, f'{image_tile_name}_mask.tif')
#     # mask_tile = mask_tile.astype(np.float32)
#     # mask_tile = mask_tile[0,:,:,:,:]
#     # mask_tile = np.moveaxis(mask_tile,0,-1)
#     # mask_tile = np.moveaxis(mask_tile,0,-1)
#     # tifffile.imsave(save_path, mask_tile)








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



# for img, mask in generate_tiles_masks_from_rois(horizontal_flip=False, vertical_flip=False, no_loop=True):
#     overlay_image_3d_mask(img, mask)



# # stack = images[:]
# fusion = mosaic.get_fusion()
# print('Fusion shape:', fusion.shape)