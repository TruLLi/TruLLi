from pylibdmtx.pylibdmtx import decode
import cv2
import numpy as np

capture = cv2.VideoCapture(0)
capture.set(3, 640)  # id za width je 3 i to je postavljeno na 640px
capture.set(4, 480)  # id za width je 4 i to je postavljeno na 480px


def dataMat(image, bgr):
    while True:
        imgCapture = capture.read()
        # gray_img = cv2.cvtColor(imgCapture, cv2.COLOR_BGR2GRAY)
        data = decode(imgCapture)
        print(data)
        for decodedObject in data:
            points = decodedObject.rect

            pts = np.array(points, np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(image, [pts], True, (0, 255, 0), 3)

            cv2.putText(frame, decodedObject.data.decode("utf-8"), (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        bgr, 2)

            print("Barcode: {} ".format(decodedObject.data.decode("utf-8")))

bgr = (8, 70, 208)

frame = cv2.imread('samples.data/dataMatrixCode.jpg')
code = dataMat(frame, bgr)
print(code)
cv2.imshow('Data Matrix reader', frame)
code = cv2.waitKey(0)