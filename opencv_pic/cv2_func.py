import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
IMAGES_FOLDER='../picresources'
earth_fname = os.path.join(IMAGES_FOLDER, 'earth.jpg')
earth_img = cv2.imread(earth_fname)
# comment out the line below to see the colour difference
earth_img = cv2.cvtColor(earth_img, cv2.COLOR_BGR2RGB)
plt.imshow(earth_img)
print('Image Shape: ', earth_img.shape, '\n\n')
print('Image Plotted:')
plt.show()

box_blur_img = earth_img.copy()
box_blur_img = cv2.blur(box_blur_img, (41, 41))
plt.imshow(box_blur_img)
plt.show()

blur_img = earth_img.copy()
blur_img = cv2.GaussianBlur(blur_img, (41, 41), 10)
plt.imshow(blur_img)
plt.show()

dilate_img = earth_img.copy()
dilate_img = cv2.dilate(dilate_img, np.ones((10,10), dtype=np.uint8), iterations=1)
plt.imshow(dilate_img)
plt.show()

erosion_img = earth_img.copy()
erosion_img = cv2.erode(erosion_img, np.ones((10,10), dtype=np.uint8), iterations=1)
plt.imshow(erosion_img)
plt.show()

canny_img = earth_img.copy()
canny_img = cv2.erode(canny_img, np.ones((12,12), dtype=np.uint8), iterations=1)
thresh = 75
edges = cv2.Canny(canny_img,thresh,thresh)
plt.imshow(edges.astype(np.uint8), cmap='gray')
plt.show()

thresh_img = earth_img.copy()
thresh_img = cv2.cvtColor(thresh_img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(thresh_img, 155, 170, cv2.THRESH_BINARY)
plt.imshow(thresh, cmap='gray')
plt.show()