import numpy as np
# from scipy.misc import imread
from imageio import imread
import matplotlib.pyplot as plt


def rgb2gray(rgb):
    b = rgb[:, :, 0].copy()
    g = rgb[:, :, 1].copy()
    r = rgb[:, :, 2].copy()
    # Gray scale
    grayimg = 0.2126 * r + 0.7152 * g + 0.0722 * b

    return grayimg.astype(np.uint8)


def sampling(img, fs):
    assert isinstance(fs, int)
    sampled_img = img[::fs, ::fs]
    return sampled_img


def quantization(img, level=1):
    assert level != 0
    assert isinstance(level, int)
    sampled_img = img
    img_size = np.shape(img)
    rang = 256
    d = round(rang / level)
    step = np.arange(0, rang, d)
    print(step)
    lenght = np.shape(step)
    for i in range(img_size[0]):
        for j in range(img_size[1]):
            for k in range(lenght[0]):
                if img[i, j] >= step[k] and img[i, j] <= step[k + 1]:
                    sampled_img[i, j] = (step[k] + step[k + 1] - 1) / 2
    return sampled_img


# Import image here
# Sample call and Plotting code
img = imread('.//picresources/windmill.png')
# img = rgb2gray(img)
print(img.shape)
plt.imshow(img, cmap='gray')
plt.show()
fs = 10
sampled_img = sampling(img, fs)
plt.imshow(sampled_img, cmap='gray')
plt.show()
level = 5
quantized_img = quantization(sampled_img, level)
plt.imshow(quantized_img, cmap='gray')
plt.show()
