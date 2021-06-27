import cv2
import os
ERODE = True
fgbg = cv2.createBackgroundSubtractorMOG2()
# fgbg = cv2.createBackgroundSubtractorKNN()
IMAGES_FOLDER='../picresources'
capture = cv2.VideoCapture(os.path.join(IMAGES_FOLDER, 'vtest.avi'))

while True:
    #     time.sleep(0.025)

    #     timer = cv2.getTickCount()

    ret, frame = capture.read()
    if frame is None:
        break

    fgMask = fgbg.apply(frame)

    cv2.rectangle(frame, (10, 2), (100, 20), (255, 255, 255), -1)
    cv2.putText(frame, str(capture.get(cv2.CAP_PROP_POS_FRAMES)), (15, 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

    cv2.imshow('Frame', frame)
    cv2.imshow('FG Mask', fgMask)

    keyboard = cv2.waitKey(30)
    if keyboard == 'q' or keyboard == 27:
        break

cv2.destroyAllWindows()
capture.release()