import cv2
import os
IMAGES_FOLDER='../picresources'
video = cv2.VideoCapture(0)

while True:
    # Read a new frame
    success, frame = video.read()
    if not success:
        # Frame not successfully read from video capture
        break

    # Display result
    cv2.imshow("frame", frame)

    k = cv2.waitKey(1) & 0xff
    if k == 27:  # escape pressed
        break
    elif k == 115:  # s pressed
        fname = input("File name")
        cv2.imwrite(os.path.join(IMAGES_FOLDER, '{}.jpg'.format(fname)), frame)

cv2.destroyAllWindows()
video.release()