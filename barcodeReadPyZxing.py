import numpy as np
import zxing
from pyzxing import BarCodeReader
import cv2


reader = BarCodeReader()
#results = reader.decode('/PATH/TO/FILE')
# Or file pattern for multiple files
#results = reader.decode('/PATH/TO/FILES/*.png')
# Or a numpy array
# Requires additional installation of opencv
# pip install opencv-python



capture = cv2.VideoCapture(0)



#reader = zxing.BarCodeReader()
#barcode = reader.decode('C:/Users/Denis/Desktop/Diplomski/pythonProject/samples.data/dataMatrixCode.jpg')

#print(barcode)

def decodeBarcodes(img):
    for barcode in reader.decode_array(img):
        print('asd', barcode)


while True:
    success, img = capture.read()
    decodeBarcodes(img)
    cv2.imshow('Image', img)
    cv2.waitKey(1)
capture.release()
cv2.destroyAllWindows()