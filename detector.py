# The detector script is run with 5 arguments: model, source, gui, input_size, num_of_frames 
# ex. python3 detector.py yolo cam gui 224 100
import numpy as np
import cv2 as cv
import os
import time
import sys
import models

confidence_threshold = 0.3
nms_threshold = 0.3
frame_width = 320
frame_height = 240
camera_width = 640
camera_height = 480

dir = os.path.dirname(__file__)

model = sys.argv[1]
cap_source = sys.argv[2]
gui = sys.argv[3]
input_size = int(sys.argv[4])
max_frames = int(sys.argv[5])

# load the model
if model == "yolo":
    net, blob_options, labels = models.YOLO()
elif model == "ssd":
    net, blob_options, labels = models.SSD()
elif model == "rcnn":
    net, blob_options, labels = models.FasterRCNN()

net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

# Create label RGB values
np.random.seed(0)
COLORS = np.random.randint(0, 255, size=(len(labels), 3), dtype="uint8")

if cap_source == "cam":
    # capture from camera
    cap = cv.VideoCapture(0)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, camera_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, camera_height)
    # number of frames to let the camera warm up before the recording starts
    wait_frames = 15
else:
    # use video file as source
    cap = cv.VideoCapture(dir+cap_source)
    wait_frames = 3
    print("Video:"+dir+cap_source)

frame_count = 0
total_inference_time = 0
start_timer = False

detections = []
scores = []
inference_times = []

def draw_boxes(indices):
    for i in indices:
        i = i[0]
        box_left = boxes[i][0]
        box_top = boxes[i][1]
        box_right = boxes[i][2]
        box_bottom = boxes[i][3]

        class_id = int(class_ids[i])
        label = labels[class_id]
        color = COLORS[class_id].tolist()
        confidence = confidences[i]
        
        #detections.append(label)
        #scores.append(confidence)        
        # only person objects are detected
        if label == "person" and start_timer:
            detections.append(label)
            scores.append(confidence)
        else:
            continue

        label = label + " " + str(round(confidence * 100, 2)) + "%"
        print(label)
        cv.rectangle(frame, (box_left, box_top), (box_right, box_bottom), color, 2)
        cv.putText(frame, label, (box_left + 5, box_top + 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def get_detections_yolo(net):
    layer_names = net.getLayerNames()
    layer_names = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    inference_time = time.time()
    layer_outputs = net.forward(layer_names)
    inference_time = time.time() - inference_time

    class_ids = []
    confidences = []
    boxes = []

    # output: 0 to 3 = center_x, center_y, width, height of bounding box, 4 and up are class probabilities
    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > confidence_threshold:
                center_x = int(detection[0] * frame.shape[1])
                center_y = int(detection[1] * frame.shape[0])
                box_width = int(detection[2] * frame.shape[1])
                box_height = int(detection[3] * frame.shape[0])
                x = int(center_x - box_width / 2)
                y = int(center_y - box_height / 2)
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, box_width + x, box_height + y])

    return class_ids, confidences, boxes, inference_time


def get_detections(net):
    inference_time = time.time()
    outputs = net.forward()
    inference_time = time.time() - inference_time

    class_ids = []
    confidences = []
    boxes = []
    # output: 1 is class id, 2 is confidence, 3-6 are left, top, right, bottom of bounding boxes
    for detection in outputs[0, 0]:

        class_id = int(detection[1])
        confidence = float(detection[2])
        if confidence > confidence_threshold:
            box_left = int(detection[3] * frame.shape[1])
            box_top = int(detection[4] * frame.shape[0])
            box_right = int(detection[5] * frame.shape[1])
            box_bottom = int(detection[6] * frame.shape[0])
            confidences.append(float(confidence))
            class_ids.append(class_id)
            boxes.append([box_left, box_top, box_right, box_bottom])

    return class_ids, confidences, boxes, inference_time


while cap.isOpened():

    if frame_count == wait_frames and not start_timer:
        total_time = time.time()
        frame_count = 0
        start_timer = True
    ret, frame = cap.read()
    if not ret:      
        break
    frame_count = frame_count + 1

    # convert to 4D blob
    blob = cv.dnn.blobFromImage(frame, blob_options["scale"], (input_size, input_size), blob_options["MeanSubtraction"], swapRB=True)

    frame = cv.resize(frame, (frame_width, frame_height))
    net.setInput(blob)

    # load the model
    if model == "yolo":
        class_ids, confidences, boxes, inference_time = get_detections_yolo(net)
    else:
        class_ids, confidences, boxes, inference_time = get_detections(net)

    inference_times.append(int(round(inference_time, 3)*1000))
    # non-max suppression is applied
    indices = cv.dnn.NMSBoxes(boxes, confidences, confidence_threshold, nms_threshold)

    draw_boxes(indices)
    if gui == "gui":
        cv.imshow("Frame", frame)

    print("\nFrame count: " + str(frame_count))

    if start_timer:
        total_inference_time = total_inference_time + inference_time
        print("Inference time: " + str(round(inference_time, 3)))
        print("Average inference time: " + str(round((total_inference_time / frame_count), 3)) + "s")
        print("Average frame time: " + str(round(((time.time() - total_time) / frame_count), 3)) + "s")
        print("Average frame rate: " + str(round(frame_count / (time.time() - total_time), 3)))

    if cv.waitKey(1) >= 0 or frame_count >= max_frames:
        total_time = time.time() - total_time 
        print("\nResults for: " + model + " " + str(input_size))
        print("Total time : " + str(round(total_time, 3)) + "s")
        print("Average inference time: " + str(round((total_inference_time / frame_count), 3)) + "s")
        print("Average time per frame: " + str(round((total_time / frame_count), 3)) + "s")
        print("Average frame rate: " + str(round(frame_count / total_time, 2)))
        print("Total Frames: " + str(frame_count))

        inference_times = ' '.join(map(str, inference_times)) 
        scores = ' '.join(map(str, scores)) 
        
        file = open("results.dat","a") 
        file.write("Results for: " + model + " " + str(input_size) + "\n")
        file.write("Average frame rate: " + str(round(frame_count / total_time, 2)) + "\n")
        file.write("Total Frames: " + str(frame_count) + "\n")
        file.write("Average inference time: " + str(round((total_inference_time / frame_count), 3)*1000) + "ms" + "\n")
        file.write("Inference times (ms): " + inference_times + "\n")
        file.write("Detection score: " + scores + "\n")
        file.close()
        break

cap.release()
cv.destroyAllWindows()
