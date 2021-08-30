import cv2
from dbr import *

capture = cv2.VideoCapture('samples.data/test1.mp4')

reader = BarcodeReader()
license_key = "t0068fQAAAHUCSUdIj65SoZfk8tKFuAjjt4DMH/W2hC/1wfgmChAn6p2ymFRrrMg+tX4sV65tWvHppcbRA9K1njK3re6G4Tg="
reader.init_license(license_key)
color = (0, 0, 255)
thickness = 2

def readBarcode(img):
    textResults = reader.decode_buffer(img);

    if (textResults is not None):
        for out in textResults:
            print(out.barcode_text)
            print(out.localization_result.localization_points)
            points = out.localization_result.localization_points
            cv2.line(img, points[0], points[1], color, thickness)
            cv2.line(img, points[1], points[2], color, thickness)
            cv2.line(img, points[2], points[3], color, thickness)
            cv2.line(img, points[3], points[0], color, thickness)
            cv2.putText(img, out.barcode_text, (min([point[0] for point in points]), min([point[1] for point in points])), cv2.FONT_HERSHEY_SIMPLEX, 1, color, thickness)

while True:
    success, img = capture.read()
    readBarcode(img)
    cv2.imshow('Image', img)
    cv2.waitKey(1)