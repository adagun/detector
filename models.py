import cv2 as cv
import os


def YOLO():

    dir = os.path.dirname(__file__)
    net = cv.dnn.readNetFromDarknet(dir + "/models/yolov3-tiny.cfg", dir + "/models/yolov3-tiny.weights")
    blob_options = {"scale": 1/255.0, "MeanSubtraction": (0, 0, 0)}
    labels = open(dir + "/data/coco2014.names").read().strip().split("\n")
    return net, blob_options, labels


def SSD():

    dir = os.path.dirname(__file__)
    net = cv.dnn.readNetFromTensorflow(dir + "/models/ssdlite_mobilenet_v2.pb", dir + "/models/ssdlite_mobilenet_v2.pbtxt")

    blob_options = {"scale": 1.0, "MeanSubtraction": (127.5, 127.5, 127.5)}

    labels = open(dir + "/data/coco2017.names").read().strip().split("\n")
    labels.insert(0, "unknown")
    return net, blob_options, labels


def FasterRCNN():

    dir = os.path.dirname(__file__)
    net = cv.dnn.readNetFromTensorflow(dir + "/models/faster_rcnn_inception_v2.pb", dir + "/models/faster_rcnn_inception_v2.pbtxt")

    blob_options = {"scale": 1, "MeanSubtraction": (127.5, 127.5, 127.5)}
    labels = open(dir + "/data/coco2017.names").read().strip().split("\n")
    return net, blob_options, labels

