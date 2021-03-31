## Parameters for training Unet-3d spotfinding 

### Project Name
The project name will be used in log files, tensorboard, and model file names<br>
project_name: richard_test

### Location of the training images
- Please use the linux path: /n/core/micro...
- Images can be any size (greater than the tile size)

train_folder: /n/core/micro/yada-yada/train

### Location of the validation images
- Please use the linux path: /n/core/micro...
- Images can be any size (greater than the tile size)

```
validation_folder: /n/core/micro/yada-yada/validate
```

### Set the extension of the image file (ex: .tif).
```
image_extension: .tif
```

### Specify the channel to train on (one indexed).
- The current network only accepts a single channel as input.

```
channel: 1
```

### Tile size is a list of x, y, z dimensions.
The input image will be broken into tiles with shape tile_size.
Tile size should be as big or bigger than the features you wish 
to find in the image. Safe values for x and y are 64, 128, 256, 
and 512. Safe values for z are 8, 16, and 32.

```
tile_size_x: 128 
tile_size_y: 128 
tile_size_z: 32 
```
### Force Z
Force the z-dimension of the input images to a specific number of slices.
The image will be cropped or padded in z accordingly.
Safe values are 8, 16, 32, and 64. 

- Example 1: If the tile_size z-dimension is 8 and the z-overlap is 0.5, then 
any multiple of 4 (above 8) is safe.
- Example 2: If the tile_size z-dimension is 16 and the z-overlap is 0.5, then 
any multiple of 8 (above 16) is safe.

```
force_z: 32
```

## Advanced paramters

### Batch Size
Any batch size is acceptable, but good choices are 8, 16, or 32.
If you run out of memory due to large z-stacks, decrease batch size.
If you have enough training data, a batch of 32 is optimal.
```
batch_size: 16
```

### Number of epochs
Number of epochs is how many consecutive times the neural net will train on the data.
Most training sessions seem to finish between 100 and 300 epochs.
```
epochs: 300
```

### Overlap fraction
Overlap fraction is a list in x, y, z dimensions.
A number between 0 and 1 representing the fraction of the tile_size 
which should overlap in each dimension when tiling the input image.
```
x overlap fraction: 0.5
y overlap fraction: 0.5
z overlap fraction: 0.5
```