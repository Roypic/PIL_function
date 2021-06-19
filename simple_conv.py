import numpy as np
# from scipy.misc import imread
from imageio import imread
import matplotlib.pyplot as plt
import time


def imkernel(tau, s1, s2):
    w = lambda i, j: np.exp(-(i ** 2 + j ** 2) / (2 * tau ** 2))
    # normalization
    i, j = np.mgrid[-s1:s1 + 1, -s2:s2 + 1]
    Z = np.sum(w(i, j))
    print(w(i, j) / Z)
    nu = lambda i, j: w(i, j) / Z * (np.absolute(i) <= s1 & np.absolute(j) <= s2)
    return nu


def imshift(x, k, l):
    # periodical boundry
    [n1, n2] = np.shape(x)
    # print(np.shape(x))
    irange = np.remainder(np.arange(n1) + k, n1)
    ipos = np.where(irange == 0)[0][0]  # vertical direction

    jrange = np.remainder(np.arange(n2) + l, n2)
    jpos = np.where(jrange == 0)[0][0]  # Horizontal direction

    temp_shifted = np.zeros([n1, n2])
    xshifted = np.zeros([n1, n2])

    if ipos != 0:
        temp_shifted[ipos:n1, :] = x[0:k, :]
        temp_shifted[0:ipos, :] = x[k:n1, :]
    else:
        temp_shifted = x
    if jpos != 0:
        xshifted[:, jpos:n2] = temp_shifted[:, 0:l]
        xshifted[:, 0:jpos] = temp_shifted[:, l:n2]
    else:
        xshifted = temp_shifted
    return xshifted


# Create imconvolve_naive function
def imconvolve_naive(x, nu, s1, s2):
    n1, n2 = np.shape(x)
    xconv = np.zeros((n1, n2))
    for i in range(s1, n1 - s1):
        for j in range(s2, n2 - s2):
            for k in range(-s1, s1 + 1):
                for l in range(-s2, s2 + 1):
                    xconv[i, j] = xconv[i, j] + nu(k, l) * x[i - k, j - l]
    return xconv


# Create imconvolve_spatial function
def imconvolve_spatial(x, nu, s1, s2):
    n1, n2 = np.shape(x)
    xconv = np.zeros((n1, n2))

    for k in range(-s1, s1 + 1):
        for l in range(-s2, s2 + 1):
            xshift = imshift(x, -k, -l)
            xconv = xconv + nu(k, l) * xshift
    return xconv


# Sample call and Plotting code
tau = 1
s1 = 2
s2 = 2
wind = imread('./picresources/windmill.png').astype(float)
lake = imread('./picresources/lake.png').astype(float)

# plt.subplot(2,2,1)
plt.imshow(wind.astype(np.uint8), cmap='gray')
plt.show()

nu = imkernel(tau, s1, s1);
print(nu(-1, -1))

t = time.time()
xconv = imconvolve_naive(wind, nu, s1, s2)
# plt.subplot(2,2,2)
plt.imshow(xconv.astype(np.uint8), cmap='gray')
elapsed = time.time() - t
print(elapsed)
plt.show()

t = time.time()
xconv2 = imconvolve_spatial(wind, nu, s1, s2)
plt.imshow(xconv2.astype(np.uint8), cmap='gray')
elapsed = time.time() - t
print(elapsed)
plt.show()