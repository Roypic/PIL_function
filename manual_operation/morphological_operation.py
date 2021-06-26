#Function
'''
Your code here
'''
import numpy as np
from imageio import imread
import matplotlib.pyplot as plt
def noise(img, perc):
    (n1, n2) = img.shape
    salt_ammount = np.ceil(n1*n2*(perc/100)/2)
    pepper_amount = np.ceil(n1*n2*(perc/100)/2)
    noisy_image = img.copy()
    for i in range(int(salt_ammount)):
        x = np.random.randint(0, n1)
        y = np.random.randint(0, n2)
        noisy_image[x, y] = 255
    for j in range(int(pepper_amount)):
        x = np.random.randint(0, n1)
        y = np.random.randint(0, n2)
        noisy_image[x, y] = 0
    return noisy_image


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
def imosf(img, typ, s1, s2):
    (n1, n2) = img.shape
    imosf_image = np.zeros((n1, n2))
    xstack = imstack(img, s1, s2)
    for row in range(n1):
            for col in range(n2):
                if typ == 'dialate':
                    imosf_image[row, col] = np.max(xstack[row, col, :])

                if typ == 'erode':
                    imosf_image[row, col] = np.min(xstack[row, col, :])

                if typ == 'median':
                    sort_array = np.sort(xstack[row, col, :], axis=0)
                    middle_value = sort_array[int(((2 * s1 + 1) * (2 * s2 + 1) - 1) / 2)]
                    imosf_image[row, col] = middle_value

                if typ == 'trimmed':
                    sort_array = np.sort(xstack[row, col, :], axis=0)
                    sort_array = sort_array[int(((2 * s1 + 1)*(2 * s2 + 1)-1)/2/4) : int((2 * s1 + 1)*(2 * s2 + 1)-((2 * s1 + 1)*(2 * s2 + 1)-1)/2/4)]
                    mean_value = np.sum(sort_array)/np.shape(sort_array)[0]
                    imosf_image[row, col] = mean_value
    return imosf_image

def imopening(img, s1, s2):
    imopening_image = imosf(img, 'dialate', s1, s2)
    imopening_image = imosf(imopening_image, 'erode', s1, s2)
    return imopening_image

def imclosing(img, s1, s2):
    imclosing_image = imosf(img, 'erode', s1, s2)
    imclosing_image = imosf(imclosing_image, 'dialate', s1, s2)
    return imclosing_image

img = imread('.//picresources//castle.png')
s1 = 2
s2 = 2
noisy_image =img# noise(img, 10)
immedian_image = imosf(noisy_image, 'median', s1, s2)
imtrimmed_image = imosf(noisy_image, 'trimmed', s1, s2)
imopening_image = imopening(noisy_image, s1, s2)
imclosing_image = imclosing(noisy_image, s1, s2)
plt.subplot(1, 5, 1)
plt.imshow(noisy_image, cmap="gray")
plt.axis('off')
plt.subplot(1, 5, 2)
plt.imshow(immedian_image, cmap="gray")
plt.axis('off')
plt.subplot(1, 5, 3)
plt.imshow(imtrimmed_image, cmap="gray")
plt.axis('off')
plt.subplot(1, 5, 4)
plt.imshow(imclosing_image, cmap="gray")
plt.axis('off')
plt.subplot(1, 5, 5)
plt.imshow(imopening_image, cmap="gray")
plt.axis('off')
plt.show()

'''
Your code here
'''
