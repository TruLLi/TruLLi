# pythonDiplomski
Config files for my GitHub profile.



- Main program is called program.py where is object detection and barcode reader united as one.
- barcodeRead.py is only for reading barcodes from camera or picture.
- objectDetection.py is only for detecting object from camera or from video.

In model.data you need to put yolo4-helmet-detection.weight file which can be downloaded for here:
https://drive.google.com/file/d/1h-Ro2EA363bNQIP6GaQNQhI0zP-5fxMK/view?usp=sharing

If you want to run on normal yolov4.weight file: 
- you need to change cfg to use yolov4.cfg
- use coco.names data set.
- Normal yolov4 weight can be downloaded here: https://drive.google.com/file/d/1St_V0EcCTNc85VEvxCHrgeDjxZpWu4kS/view?usp=sharing 

In samples.data folder you can find basic videos and photos which on you can test on.

