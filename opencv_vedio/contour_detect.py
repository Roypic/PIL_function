import cv2
import numpy as np
IMAGES_FOLDER='../picresources'
import os
OUTLINE = True
LRG_ONLY = True

# window to hold the trackbar
img = np.zeros((300, 512, 3), np.uint8)
cv2.namedWindow('image')

# create trackbar
cv2.createTrackbar('Thresh', 'image', 0, 255, lambda x: None)

earth_fname = os.path.join(IMAGES_FOLDER, 'earth.jpg')
earth_img = cv2.imread(earth_fname)

while True:
    thresh_min = cv2.getTrackbarPos('Thresh', 'image')

    contour_img = earth_img.copy()
    contour_img = cv2.cvtColor(contour_img, cv2.COLOR_BGR2GRAY)
    ret, contour_img_thresh = cv2.threshold(contour_img, thresh_min, 255, 0)
    contours, hierarchy = cv2.findContours(contour_img_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if LRG_ONLY:
        cnts = [x for x in contours if cv2.contourArea(x) > 20000]
    else:
        cnts = sorted(contours, key=cv2.contourArea, reverse=True)[:3]

    if OUTLINE:
        # Draw only outlines
        contour_img_display = cv2.drawContours(earth_img.copy(), cnts, -1, (238, 255, 0), 2)
    else:
        # Draw filled contours
        contour_img_display = cv2.drawContours(earth_img.copy(), cnts, -1, (238, 255, 0), -1)

    contour_img_display = cv2.cvtColor(contour_img_display, cv2.COLOR_BGR2RGB)

    cv2.imshow('image', contour_img_display)
    cv2.imshow('thresh', contour_img_thresh)

    k = cv2.waitKey(1) & 0xff
    if k == 27: break  # ESC pressed

cv2.destroyAllWindows()
# plot_image(contour_img_thresh)
# plot_image(contour_img_display, recolour=True) 