import cv2
import numpy as np

import time
import sys
import os

from numpy.lib.type_check import imag
boxes, confidences, class_ids = [], [], []
def Phase1(image):
    h, w = image.shape[:2]
    # create 4D blob
    blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
    print("image.shape:", image.shape)
    print("blob.shape:", blob.shape)
    # sets the blob as the input of the network
    net.setInput(blob)
    # get all the layer names
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    # feed forward (inference) and get the network output
    # measure how much it took in seconds
    start = time.perf_counter()
    layer_outputs = net.forward(ln)
    time_took = time.perf_counter() - start
    print(f"Time took: {time_took:.2f}s")
    font_scale = 0.5
    thickness = 1

    # loop over each of the layer outputs
    for output in layer_outputs:
        # loop over each of the object detections
        for detection in output:
            # extract the class id (label) and confidence (as a probability) of
            # the current object detection
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            # discard out weak predictions by ensuring the detected
            # probability is greater than the minimum probability
            if confidence > CONFIDENCE:
                # scale the bounding box coordinates back relative to the
                # size of the image, keeping in mind that YOLO actually
                # returns the center (x, y)-coordinates of the bounding
                # box followed by the boxes' width and height
                box = detection[:4] * np.array([w, h, w, h])
                (centerX, centerY, width, height) = box.astype("int")
                # use the center (x, y)-coordinates to derive the top and
                # and left corner of the bounding box
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                # update our list of bounding box coordinates, confidences,
                # and class IDs
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    print(detection.shape)
    # perform the non maximum suppression given the scores defined before
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, SCORE_THRESHOLD, IOU_THRESHOLD)
    # ensure at least one detection exists
    if len(idxs) > 0:
        # loop over the indexes we are keeping
        for i in idxs.flatten():
            # extract the bounding box coordinates
            x, y = boxes[i][0], boxes[i][1]
            w, h = boxes[i][2], boxes[i][3]
            # draw a bounding box rectangle and label on the image
            color = [int(c) for c in colors[class_ids[i]]]
            cv2.rectangle(image, (x, y), (x + w, y + h), color=color, thickness=thickness)
            text = f"{labels[class_ids[i]]}"
            # calculate text width & height to draw the transparent boxes as background of the text
            (text_width, text_height) = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fontScale=font_scale, thickness=thickness)[0]
            text_offset_x = x
            text_offset_y = y - 5
            box_coords = ((text_offset_x, text_offset_y), (text_offset_x + text_width + 2, text_offset_y - text_height))
            overlay = image.copy()
            cv2.rectangle(overlay, box_coords[0], box_coords[1], color=color, thickness=cv2.FILLED)
            # add opacity (transparency to the box)
            image = cv2.addWeighted(overlay, 0.6, image, 0.4, 0)
            # now put the text (label: confidence %)
            cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=font_scale, color=(0, 0, 0), thickness=thickness)

#    cv2.imshow("yolov5", image)
    #waits for user to press any key 
    #(this is necessary to avoid Python kernel form crashing)
#    cv2.waitKey(0) 
    
    #closing all open windows 
#    cv2.destroyAllWindows()
    return idxs 
def Phase2(idxs):
    def x(e):
        return e[0]

    def y(e):
        return e[1]
        
    data = []
    vis = []


    for i in idxs.flatten():
        object = [boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3], labels[class_ids[i]]]
        data.append(object)
        vis.append(0)


    data.sort(key=x)
    levels = []
    i = 0
    while i < len(data):
        object1 = []
        ans=0
        max = int(data[i][0] + (data[i][2] / 2))
        for j in range(len(data)):
            if data[j][0] <= max and vis[j] == 0:
                object1.append(data[j])
                vis[j] = 1
                i += 1
        object1.sort(reverse=True, key=y)
        
        levels.append(object1)

#    for i in range(len(levels)):
#        print(levels[i])
    return levels

CONFIDENCE = 0.5
SCORE_THRESHOLD = 0.5
IOU_THRESHOLD = 0.5

# the neural network configuration
config_path = "C:/Users/Laptop Shop/Desktop/custom-yolov4-detector.cfg"
# the YOLO net weights file
weights_path = "C:/Users/Laptop Shop/Desktop/custom-yolov4-detector_best.weights"
# weights_path = "weights/yolov3-tiny.weights"

# loading all the class labels (objects)
labels = open("C:/Users/Laptop Shop/Desktop/coco.names").read().strip().split("\n")
# generating colors for each object for later plotting
colors = np.random.randint(0, 255, size=(len(labels), 3), dtype="uint8")
# load the YOLO network
net = cv2.dnn.readNetFromDarknet(config_path, weights_path)

path_name = "C:/Users/Laptop Shop/Desktop/source.jpeg"
reference = cv2.imread(path_name)
Ref1 = cv2.cvtColor(reference.copy(), cv2.COLOR_BGR2RGB)
Ridxs=Phase1(reference)
rlevels=Phase2(Ridxs)
boxes, confidences, class_ids = [], [], []
# #
image = cv2.imread("C:/Users/Laptop Shop/Desktop/dest.jpeg")
image_copy=cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
idxs=Phase1(image)
ilevels=Phase2(idxs)

if len(ilevels) != len(rlevels):
    if abs(len(ilevels[0]) - len(rlevels[0])) > 2:
        if len(ilevels) > len(rlevels):
            rlevels.insert(0, [])
        else:
            ilevels.insert(0, [])
    else:
        if abs(len(ilevels[len(ilevels)-1]) - len(rlevels[(len(rlevels)-1)])) > 3:
            rlevels.append([])
        else:
            ilevels.append([])



  

print("--------------------The Reference data:-----------------\n")
for i in range(len(rlevels)):
    print(len(rlevels[i]), " level:",i," ",rlevels[i])
print("--------------------The image data:-----------------\n")
for i in range (len(ilevels)):
    print(len(ilevels[i]), " level:",i," ",ilevels[i])
# change to pink blue 4 to white red 3  damage with yellow 2 growth green 1
spot_d=[]
maxlen = max(len(rlevels),len(ilevels))
# get the minmum for each level in  l1 and l2 
# for example 6 check each elment if change blue or red 
# change return 4 for blue and 3 for red  and 0 for equal 

#ilevel , rlevel
#elbow: 4, 5 
#t: 1, 0
#pipe: 2, 3
growth = []
damage = []
ptw = []
wtp = []



def compare(arr1, arr2):
    for i in range(len(arr1)):
        if (arr1[i][4]== '1' and arr2[i][4]=='0') or (arr1[i][4]== '2' and arr2[i][4]=='3') or (arr1[i][4]== '4' and arr2[i][4]=='5'):
            wtp.append(arr1[i])
        elif (arr1[i][4]== '0' and arr2[i][4]=='1') or (arr1[i][4]== '3' and arr2[i][4]=='2') or (arr1[i][4]== '5' and arr2[i][4]=='4'):
            ptw.append(arr1[i])

for i in range(len(ilevels)):
    if ilevels[i] == [] or rlevels[i] == []:
        continue
    diff = 0
    if (len(ilevels[i]) == len(rlevels[i])):
        compare(ilevels[i], rlevels[i])
    elif len(ilevels[i]) > len(rlevels[i]):
        diff = len(ilevels[i]) - len(rlevels[i])
        for j in range(len(rlevels[i])):
            if int(rlevels[i][j][4])//2 != int(ilevels[i][j][4])//2:
                growth.append(ilevels[i][j])
                del(ilevels[i][j])
                j -= 1
                diff -= 1
                if diff == 0:
                    break
        if diff > 0:
            for j in range(diff):
                growth.append(ilevels[i][len(rlevels[i])])
                del(ilevels[i][len(rlevels[i])])

        compare(ilevels[i], rlevels[i])
    else:
        diff = len(rlevels[i]) - len(ilevels[i])
        for j in range(len(ilevels[i])):
            if int(rlevels[i][j][4])//2 != int(ilevels[i][j][4])//2:
                damage.append(rlevels[i][j])
                del(rlevels[i][j])
                j -= 1
                diff -= 1
                if diff == 0:
                    break
        if diff > 0:
            for j in range(diff):
                damage.append(rlevels[i][len(ilevels[i])])
                del(rlevels[i][len(ilevels[i])])
        compare(ilevels[i], rlevels[i])

    # draw a bounding box rectangle and label on the image
destimg=image.copy()
for i in range(len(wtp)):
    cv2.rectangle(image, (wtp[i][0], wtp[i][1]), (wtp[i][0] + wtp[i][2], wtp[i][1] + wtp[i][3]), color=(255,0,0), thickness=1)

for i in range(len(ptw)):
    cv2.rectangle(image, (ptw[i][0], ptw[i][1]), (ptw[i][0] + ptw[i][2], ptw[i][1] + ptw[i][3]), color=(0,0,255), thickness=1)

for i in range(len(growth)):
    cv2.rectangle(image, (growth[i][0], growth[i][1]), (growth[i][0] + growth[i][2], growth[i][1] + growth[i][3]), color=(0,255,0), thickness=1)

for i in range(len(damage)):
    cv2.rectangle(image, (damage[i][0], damage[i][1]), (damage[i][0] + damage[i][2], damage[i][1] + damage[i][3]), color=(0,255,255), thickness=1)
# calculate text width & height to draw the transparent boxes as background of the text
#cv2.imshow("res", image)
#waits for user to press any key 

import numpy as np
import matplotlib.pyplot as plt

all_imgz = [Ref1, image_copy, cv2.cvtColor(image, cv2.COLOR_BGR2RGB)]
titles = ["Reference", "Current", "Differences"]
import matplotlib.patches as mpatches

b_patch = mpatches.Patch(color='blue', label='Change from White to Pink')
g_patch = mpatches.Patch(color='green', label='Growth')
r_patch = mpatches.Patch(color='red', label='Change from Pink to White')
y_patch = mpatches.Patch(color='yellow', label='Damage')
w=10
h=10
fig=plt.figure(figsize=(8, 8))
columns = 3
rows = 1
for i in range(1, columns*rows +1):
    img = all_imgz[i-1]
    fig.add_subplot(rows, columns, i)
    plt.title(titles[i-1])
    plt.imshow(img)
fig.legend(handles=[b_patch, g_patch, r_patch, y_patch])
plt.show()

#(this is necessary to avoid Python kernel form crashing)
# Hori = np.concatenate((reference, destimg,image), axis=1)

cv2.waitKey(0)  

#closing all open windows 
cv2.destroyAllWindows()

print(len(wtp))
print(len(ptw))
print(len(growth))
print(len(damage))

