import cv2
import numpy as np
from pyzbar.pyzbar import decode as qrCodeDecode
from pylibdmtx.pylibdmtx import decode as dmCodeDecode

qr = qrCodeDecode(cv2.imread('frame.png'))
dx = dmCodeDecode(cv2.imread('dataMatrixCode.jpg'))
print(dx)
print(qr)

# dodavanje kamere
capture = cv2.VideoCapture(0)
capture.set(3, 640)  # id za width je 3 i to je postavljeno na 640px
capture.set(4, 480)  # id za width je 4 i to je postavljeno na 480px

while True:
    success, imgCapture = capture.read()
    for barcode in qrCodeDecode(imgCapture):
        print(barcode.data)
        myData = barcode.data.decode('utf-8')  # generalna metoda za decode sto pretvara vrijednost barcoda u string
        print(myData)
        # dodavanje bounding boxeva

        points = np.array([barcode.polygon], np.int32)
        points = points.reshape((-1, 1, 2))  # pozicioniranje obruba oko barcoda
        cv2.polylines(imgCapture, [points], True, (0, 255, 106), 4)  # dodavanje pointova oko barcodova kada ih procita
        textPoints = barcode.rect  # ovo su pointovi za text kada procita barcode
        cv2.putText(imgCapture, myData, (textPoints[0], textPoints[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 217, 255), 2)
    cv2.imshow('Result', imgCapture)
    cv2.waitKey(1)  # delay od 1ms

    #ovo je test
