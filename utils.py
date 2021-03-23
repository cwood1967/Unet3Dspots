import matplotlib.pyplot as plt
import numpy as np

def show_image(image):
    print(image.shape)
    fig = plt.figure(figsize=(16, 16))
    plt.imshow(image, cmap="gray")
    # plt.axis("off")   # turns off axes
    plt.axis("tight")  # gets rid of white border
    plt.axis("image")  # square up the image instead of filling the "figure" space
    plt.show()

def show_image_3d(image):
    print(image.shape)
    #Remove batch and channel dimensions
    if len(image.shape)==5:
        image = image[0,:,:,:,0]
    # Remove channel dimension
    if len(image.shape)==4:
        image = image[:,:,:,0]
    # Z-project: Axes order is y,x,z
    image = np.max(image, axis=2)
    fig = plt.figure(figsize=(16, 16))
    plt.imshow(image, cmap="gray")
    # plt.axis("off")   # turns off axes
    plt.axis("tight")  # gets rid of white border
    plt.axis("image")  # square up the image instead of filling the "figure" space
    plt.show()

def overlay_image_3d_mask(image, mask):
    print(image.shape)
    #Remove batch and channel dimensions
    if len(image.shape)==5:
        image = image[0,:,:,:,0]
    if len(mask.shape)==5:
        mask = mask[0,:,:,0,0]
    # Remove channel dimension
    if len(image.shape)==4:
        image = image[:,:,:,0]
    if len(mask.shape)==4:
        mask = mask[:,:,0,0]
    # Z-project: Axes order is y,x,z
    image = np.max(image, axis=2)
    fig = plt.figure(figsize=(16, 16))
    plt.imshow(image, cmap="gray")
    plt.imshow(mask, cmap='Greens', alpha=0.8)
    # plt.axis("off")   # turns off axes
    plt.axis("tight")  # gets rid of white border
    plt.axis("image")  # square up the image instead of filling the "figure" space
    plt.show()