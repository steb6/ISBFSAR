import cv2

index = 0
arr = []
while True:
    cap = cv2.VideoCapture(index)
    if cap.read()[0]:
        arr.append(index)
        cap.release()
    index += 1
    if index == 10:
        break
print(arr)
exit()