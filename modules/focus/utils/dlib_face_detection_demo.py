import dlib
import cv2

bbox_detector = dlib.get_frontal_face_detector()
cam = cv2.VideoCapture(0)

while True:
    ret, img = cam.read()
    res = bbox_detector(img)
    if len(res) > 0:
        res = res[0]
        img = cv2.rectangle(img, (res.left(), res.top()), (res.right(), res.bottom()), (255, 0, 0), 2)
    cv2.imshow("", img)
    cv2.waitKey(1)
