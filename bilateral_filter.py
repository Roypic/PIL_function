#Function
'''
Your code here
'''
import random
import numpy as np
from imageio import imread
import matplotlib.pyplot as plt
from scipy import signal
#note: noise can be generated using
# noise: sigma*np.random.randn(n1,n2)
def GaussianNoise(img, means, sigma):
    n1, n2 = img.shape
    noise_img = np.zeros((n1, n2))
    for i in range(n1):
        for j in range(n2):
            noise_img[i, j] = img[i, j] + random.gauss(means, sigma)
            if noise_img[i, j] < 0:
                noise_img[i, j] = 0
            elif noise_img[i, j] > 255:
                noise_img[i, j] = 255

    return noise_img

def imstack(img,s1,s2):
    (n1, n2) = img.shape
    s = (2*s1 + 1) * (2*s2 + 1)
    xstack = np.zeros((n1, n2, s))
    for i in range(s1, n1 - s1):
        for j in range(s2, n2 - s2):
            image_block = img[i - s1:i + s1 + 1, j - s2:j + s2 + 1]
            for a in range(2 * s1 + 1):
                for b in range(2 * s2 + 1):
                    xstack[i, j, (2 * s1 + 1) * a + b] = image_block[a, b]

    return xstack

def imkernel_space(tau, s1, s2):
    w = lambda i, j: np.exp(-(i**2 + j**2)/(2*tau**2))
    i, j = np.mgrid[-s1:s1 + 1, -s2:s2 + 1]
    nu = lambda i, j: w(i, j)

    return nu

def imkernel_value(central_value, amount, tau, s1, s2, xstack, i, j):
    w = lambda i: np.exp(-(i ** 2)/(2*tau**2))
    xstack_segment = np.zeros(amount)
    for t in range(amount):
        xstack_segment[t] = w((central_value-xstack[i, j, t]))
    xstack_segment_anoform = np.zeros((2 * s1 + 1, 2 * s2 + 1))
    for a in range(2 * s1 + 1):
        for b in range(2 * s2 + 1):
            xstack_segment_anoform[a, b] = xstack_segment[a * (2 * s1 + 1) + b]

    return xstack_segment_anoform

def imkernel_mul(xstack_segment_anoform, nu, s1, s2):
    mul_coefficient = np.zeros((2 * s1 + 1, 2 * s2 + 1))
    k = -s1
    for a in range(2 * s1 + 1):
        l = -s2
        for b in range(2 * s2 + 1):
            mul_coefficient[a, b] = xstack_segment_anoform[a, b] * nu(k, l)
            l = l + 1
        k = k + 1
    sum = np.sum(mul_coefficient)
    mul_coefficient = mul_coefficient/sum

    return mul_coefficient

def imbilateral_naive(x, xstack, nu, sigma, s1, s2):
    n1, n2 = np.shape(x)
    xconv = np.zeros((n1, n2))
    m1, m2, m3 = np.shape(xstack)
    for i in range(s1, n1 - s1):
        for j in range(s2, n2 - s2):
            central_value = x[i, j]
            xstack_segment_anoform = imkernel_value(central_value, m3, sigma, s1, s2, xstack, i, j)
            mul_coefficient = imkernel_mul(xstack_segment_anoform, nu, s1, s2)
            k = i
            for a in range(2 * s1 + 1):
                l = j
                for b in range(2 * s2 + 1):
                    xconv[i, j] = xconv[i, j] + mul_coefficient[a, b] * x[k - s1, l - s2]
                    l = l + 1
                k = k + 1
    return xconv





img = imread('.//picresources//castle.png')
gaussnoise_img = GaussianNoise(img, 0, 10)
sigma_space = 10
sigma_value = 10
s1 = 2
s2 = 2
nu = imkernel_space(sigma_space, s1, s2)
xstack = imstack(gaussnoise_img, s1, s2)
img_processed = imbilateral_naive(gaussnoise_img, xstack, nu, sigma_value, s1, s2)
plt.subplot(1, 3, 1)
plt.imshow(img, cmap="gray")
plt.axis('off')
plt.subplot(1, 3, 2)
plt.imshow(gaussnoise_img, cmap="gray")
plt.axis('off')
plt.subplot(1, 3, 3)
plt.imshow(img_processed, cmap="gray")
plt.axis('off')
plt.show()



#Import image here
# Sample call
# castle.png
'''
Your code here
'''