import numpy as np
from imageio import imread
import matplotlib.pyplot as plt
def AHE(im, win_size):
    '''
    Your code here
    '''
    padsize = win_size // 2
    w, h = im.shape
    output = np.zeros((w , h ))
    for i in range(w):
        for j in range(h):
            winsize = win_size // 2

            rank = 0
            x1=0 if (i - winsize)<0 else i - winsize
            x2=w-1 if (i + winsize)==w else i + winsize
            y1 = 0 if (j - winsize) < 0 else j - winsize
            y2 = w - 1 if (j + winsize) == h else j + winsize
            rank = np.sum((im[x1:x2, y1:y2] <= im[i, j]).astype(int))

            output[i, j] = rank * 255 / win_size * win_size

    return output
img = imread(".//picresources//beach.png")
afterpic = AHE(img, 129)
print(afterpic.shape)
plt.figure("AHE")
plt.imshow(afterpic, cmap='gray')
plt.show()