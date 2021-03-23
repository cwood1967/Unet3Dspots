# 3D Unet 

This repository contains the DNN and data pipeline to create 3D Unet models.

Important files:

- (`data_generator_3d_tiles`)[data_generator_3d_tiles.py]: All of the tooling and data pipleine is contained in this file. The methods are shared among all of the files below to ensure consistency between training, prediction, and merging.
- (`train_3d_tiles_dash`)[train_3d_tiles_dash.py]: Builds Unet with given parameters and trains the model
- (`predict_3d_dash`)[predict_3d_dash.py]: Runs infernce and outputs tiled prediction images
- (`merge_tiles_3d_dash`)[merge_tiles_3d_dash.py]: Merges tiled predictions with the original image where prediction is in the last channel

Other files:

- (`custom_objects`)[custom_objects.py]: Custom metrics and functions for the Keras model. These are defined in a separate file so that they can be accessed both for training and inference. If a model needs to be retrained, these objects must be added before the model is compiled.
- (`utils`)[utils.py]: Some utilities for showing images in a .ipynb notebook.
- (`inference.yml`)[inference.yml]: An example yaml file containing the parameters needed to start training or inference runs.

Logs:

When a training run begins, a (`logs/`)[logs/] directory will be created in the root folder to hold all tensorboard information.