import cv2
import numpy as np

capture = cv2.VideoCapture(0)
widthAndHeight = 320
confidanceThreshHold = 0.5
nmsThreshold = 0.3

classesFile = r'C:\Users\Denis\Desktop\Diplomski\pythonProject\model.data\custom.names'
classNames = []

with open(classesFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')
    print(classNames)

modelConf = r'C:\Users\Denis\Desktop\Diplomski\pythonProject\model.data\yolov4-helmet.cfg'
modelWeig = r'C:\Users\Denis\Desktop\Diplomski\pythonProject\model.data\yolov4-helmet-detection.weights'
net = cv2.dnn.readNetFromDarknet(modelConf, modelWeig)

net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
colors = [tuple(255 * np.random.rand(3)) for i in range(5)]


def findObjects(outputs, img):
    height, width, channels = img.shape
    boundingBox = []
    classIds = []
    confidanceValue = []

    for output in outputs:
        for detection in output:
            scores = detection[5:] #ukloni ostalih pet feildova iz responsa i gledaj samo confidance value
            classId = np.argmax(scores)
            confidance = scores[classId]
            if confidance > confidanceThreshHold:
                w, h = int(detection[2] * width), int(detection[3] * height)
                x, y = int((detection[0] * width) - w/2), int((detection[1] * height) - h/2)
                boundingBox.append([x, y, w, h])
                classIds.append(classId)
                confidanceValue.append(float(confidance))

    indices = cv2.dnn.NMSBoxes(boundingBox, confidanceValue, confidanceThreshHold, nmsThreshold)

    for color, i in zip(colors, indices):
        i = i[0]
        box = boundingBox[i]
        x, y, w, h = box[0], box[1], box[2], box[3]
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        cv2.putText(img, f'{classNames[classIds[i]].upper()} {int(confidanceValue[i] * 100)}%', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

while True:
    success, img = capture.read()
    blob = cv2.dnn.blobFromImage(img, 1/255, (widthAndHeight, widthAndHeight), [0, 0, 0], 1, crop=False)
    net.setInput(blob)
    layerNames = net.getLayerNames()
    #print(layerNames)
    outPutNames = [layerNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    #print(outPutNames)
    outputs = net.forward(outPutNames)
    #print(outputs[0].shape)
    #print(outputs[1].shape)
    #print(outputs[2].shape)
    #print(outputs[0][0])
    findObjects(outputs, img)
    cv2.imshow('image', img)
    cv2.waitKey(1)
