import matplotlib.pyplot as plt
import numpy as np
from skimage.transform import rescale
from skimage.filters import threshold_otsu

import cv2


MONA_LISA = './media/mona_lisa.jpg'
CAMERAMAN = './media/cameraman.jpg'


def mona_lisa_downscaled_w_annotation(scale=0.75, **kwargs):

    f, axes = plt.subplots(1, 1, **kwargs)

    img = plt.imread(MONA_LISA)
    img = rescale(img, scale, anti_aliasing=False, multichannel=True)
    img = np.floor(255*img[:,:,0])

    axes.imshow(img, cmap='gray', vmin=0, vmax=255)
    axes.set_xlabel("pixel")
    axes.set_ylabel("pixel")

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            color = str((255 - img[i, j])/255)
            axes.text(j, i, int(img[i, j]), ha="center", va="center", color=color)

    return (f, axes)

def mona_lisa_downscaled(n=4):


    f, axes = plt.subplots(1, n, figsize=(5 + 5*(n-1),10))

    img = plt.imread(MONA_LISA)

    for ax in axes:
        
        ax.imshow(img)
        img = rescale(img, 0.25, anti_aliasing=False, multichannel=True)

        ax.set_xlabel("pixel")
        ax.set_ylabel("pixel")

    return (f, axes)

def mona_lisa_as_line(n=3):

    f, ax = plt.subplots(1, 1, figsize=(15,5))

    img = plt.imread(MONA_LISA)
    img = rescale(img, 0.25**n, anti_aliasing=False, multichannel=True)

    dims = np.shape(img)
    img = np.reshape(img, (1, dims[0]*dims[1], dims[-1]))

    ax.imshow(img)
    ax.set_xlabel("pixel")
    ax.get_yaxis().set_visible(False)

    return (f, ax)

def cameraman_scale_space(ks = [1, 5, 11, 51]):

    img = plt.imread(CAMERAMAN)
    img = img/255

    
    n = len(ks)
    f, axes = plt.subplots(1, n, figsize=(5 + 5*(n-1),10))

    for ax, k in zip(axes, ks):
        
        image = cv2.GaussianBlur(img, (k,k), 0, 0, cv2.BORDER_DEFAULT)
        ax.imshow(image, cmap='gray', vmin=0, vmax=1)
        if k != 1:
            ax.set_title(f"Filtrering på nivå av {k-1} pixlar")
        else: 
            ax.set_title(f"Ingen filtrering")

        ax.set_xlabel("pixel")
        ax.set_ylabel("pixel")

    return (f, axes)

def figure_scale_space(filename, ks = [1, 5, 11, 51]):

    img = plt.imread(filename)

    
    n = len(ks)
    f, axes = plt.subplots(1, n, figsize=(5 + 5*(n-1),10))

    for ax, k in zip(axes, ks):
        
        image = cv2.GaussianBlur(img, (k,k), 0, 0, cv2.BORDER_DEFAULT)
        ax.imshow(image, cmap='gray', vmin=0, vmax=1)
        if k != 1:
            ax.set_title(f"Filtrering på nivå av {k-1} pixlar")
        else: 
            ax.set_title(f"Ingen filtrering")

        ax.set_xlabel("pixel")
        ax.set_ylabel("pixel")

    return (f, axes)


def remove_background(filename, level=500):
    
    n = 2
    f, axes = plt.subplots(1, n, figsize=(5 + 5*(n-1),10))
    img = plt.imread(filename)
    img = img/np.amax(img)

    # img = cv2.GaussianBlur(img, (5,5), 0, 0, cv2.BORDER_DEFAULT)
    background = cv2.GaussianBlur(img, (level,level), 0, 0, cv2.BORDER_DEFAULT)

    axes[0].imshow(img, cmap='gray', vmin=0, vmax=1)
    axes[1].imshow(img-background, cmap='gray', vmin=0, vmax=1)

    for ax in axes:

        ax.set_xlabel("pixel")
        ax.set_ylabel("pixel")

    axes[0].set_title("Före filtrering")
    axes[1].set_title("Med bakgrunden subtraherad")


    return f, axes


def cameraman_w_noise(noise_type, **kwargs):

    f, ax = plt.subplots(1, 1, **kwargs)

    img = plt.imread(CAMERAMAN)
    img = img/255

    if noise_type == 'gauss':
        noise = np.random.normal(0, 0.05, img.shape)
    elif noise_type == 'salt-och-peppar':
        noise = np.random.choice((-1, 0, 1), size=img.shape, p=(0.05, 0.9, 0.05))
    elif noise_type == 'period':
        noise = np.random.normal(0, 0.05, img.shape)
        x = np.linspace(-np.pi, np.pi, img.shape[0])
        y = np.linspace(-np.pi, np.pi, img.shape[1])
    
        x, y = np.meshgrid(x, y)

        p = np.sin((x+y))

        noise = p * noise
    elif noise_type == 'kvantiserings':
        img = rescale(img, 0.25, anti_aliasing=False, multichannel=False)
        noise = np.zeros(img.shape)

    else:
        raise ValueError

    ax.imshow(np.clip(img+noise, 0, 1), cmap='gray', vmin=0, vmax=1)
    ax.set_xlabel("pixel")
    ax.set_ylabel("pixel")
    ax.set_title(f"Figur med {noise_type}-brus")

    return (f, ax)

def segment_cameraman(**kwargs):

    f, ax = plt.subplots(1, 1, **kwargs)

    img = plt.imread(CAMERAMAN)
    thresh = threshold_otsu(img)
    binary = img > thresh

    ax.imshow(binary, cmap='gray', vmin=0, vmax=1)
    ax.set_xlabel("pixel")
    ax.set_ylabel("pixel")

    return (f, ax)

def segment_threshold(filename, **kwargs):

    f, ax = plt.subplots(1, 1, **kwargs)

    img = plt.imread(filename)
    thresh = threshold_otsu(img)
    binary = img > thresh

    ax.imshow(binary, cmap='gray', vmin=0, vmax=1)
    ax.set_xlabel("pixel")
    ax.set_ylabel("pixel")

    return (f, ax)