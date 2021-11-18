import numpy as np
import cv2
import imutils
import os
import time
import random
from centroidtracker import CentroidTracker
from trackableobject import TrackableObject

ct = CentroidTracker(maxDisappeared=100, maxDistance=100)
trackers = []
trackableObjects = {}
prevTime = time.time()
Entry = 0
Exit = 0
max_people=0
people_inside=0


direction_x = 0
direction_y = 0
frame_count = 0
max_count = 0
def Check(a,  b, image):
    dist = ((a[0] - b[0]) ** 2 + 500 / ((a[1] + b[1]) / 2) * (a[1] - b[1]) ** 2) ** 0.5
    # text = "Distance:"+str(dist)
    # cv2.putText(image , text, (10, H - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    calibration = (a[1] + b[1]) / 2
    if 0 < dist < 0.25 * calibration:
        return True
    else:
        return False

def Setup(yolo):
    global net, ln, LABELS, totalExit, totalEntry
    totalEntry=0
    totalExit=0
    weights = os.path.sep.join([yolo, "yolov3.weights"])
    config = os.path.sep.join([yolo, "yolov3.cfg"])
    labelsPath = os.path.sep.join([yolo, "coco.names"])
    LABELS = open(labelsPath).read().strip().split("\n")
    net = cv2.dnn.readNetFromDarknet(config, weights)
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]


def ImageProcess(image):
    global Entry, Exit, max_people, people_inside, direction_y, direction_x, prevTime, processedImg
    global max_count
    count = 0
    rects = []
    Times = []
    (H, W) = (None, None)
    frame = image.copy()
    if W is None or H is None:
        (H, W) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    starttime = time.time()
    layerOutputs = net.forward(ln)
    stoptime = time.time()
    print("Video is Getting Processed at {:.4f} seconds per frame".format((stoptime - starttime)))
    confidences = []
    outline = []

    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            maxi_class = np.argmax(scores)
            confidence = scores[maxi_class]
            if LABELS[maxi_class] == "person":
                if confidence > 0.5:
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    outline.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))

    box_line = cv2.dnn.NMSBoxes(outline, confidences, 0.5, 0.3)

    if len(box_line) > 0:
        flat_box = box_line.flatten()
        pairs = []
        center = []
        status = []
        for i in flat_box:
            (x, y) = (outline[i][0], outline[i][1])
            (w, h) = (outline[i][2], outline[i][3])
            center.append([int(x + w / 2), int(y + h / 2)])
            status.append(False)
            Times.append(0)
        prevTime = 0
        for i in range(len(center)):
            for j in range(len(center)):
                close = Check(center[i], center[j], frame)
                if close:
                    pairs.append([center[i], center[j]])
                    Times[i] +=1
                    Times[j] +=1
                    status[i] = True
                    status[j] = True

        index = 0
        for i in flat_box:
            (x, y) = (outline[i][0], outline[i][1])
            (w, h) = (outline[i][2], outline[i][3])
            # cv2.circle(frame, (x + w // 2, y + h // 2), 2, (0, 255, 0), -1)
            if status[index] == True and Times[index] > 3:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 150), 2)
                count += 1
                if count > max_count:
                    max_count = count
            elif status[index] == False:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            else:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            rects.append((x, y, x + w, y + h))
            index += 1
        objects = ct.update(rects)
        # loop over the tracked objects
        for (objectID, centroid) in objects.items():
            # check to see if a trackable object exists for the current
            # object ID
            to = trackableObjects.get(objectID, None)

            # if there is no existing trackable object, create one
            if to is None:
                to = TrackableObject(objectID, centroid)

            # otherwise, there is a trackable object so we can utilize it
            # to determine direction
            else:
                # the difference between the y-coordinate of the *current*
                # centroid and the mean of *previous* centroids will tell
                # us in which direction the object is moving (negative for
                # 'up' and positive for 'down')
                y = [c[1] for c in to.centroids]
                x = [c[0] for c in to.centroids]
                id = to.objectID
                direction_y = centroid[1] - np.mean(y)
                direction_x = centroid[0] - np.mean(x)
                to.centroids.append(centroid)
                # check to see if the object has been counted or not
                if not to.counted:
                    # if the direction is negative (indicating the object
                    # is moving up) AND the centroid is above the center
                    # line, count the object
                    if axis == "H":
                        if centroid[1] < int(L_pos):
                            people_inside+=1
                            to.counted = True
                            print("ID", id, " inside")
                            print("people inside: ",people_inside)
                        if int(people_inside-Exit) >= int(max_people):
                            cv2.putText(frame, "No More People Allowed", (450, H - 360), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                        if direction_y < 0 and centroid[1] < int(L_pos) and centroid[1] > int(L_pos)-10 :
                            Entry = Entry+1
                            to.counted = True
                            print(id, " Entering")
                        # if the direction is positive (indicating the object
                        # is moving down) AND the centroid is below the
                        # center line, count the object
                        elif direction_y > 0 and centroid[1] > int(L_pos) and centroid[1] < int(L_pos)+10 :
                            Exit = Exit+1
                            to.counted = True
                            print(id, " Exiting")
                    if axis == "V":
                        if centroid[0] < int(L_pos):
                            people_inside+=1
                            to.counted = True
                            print("ID", id, " inside")
                            print("people inside: ",people_inside)
                        if int(people_inside-Exit) >= int(max_people):
                            cv2.putText(frame, "No More People Allowed", (450, H - 360), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                        if direction_x < 0 and centroid[0] < int(L_pos) and centroid[0] > int(L_pos)-10:
                            Entry += 1
                            to.counted = True
                            print(id, " Entering")
                        # if the direction is positive (indicating the object
                        # is moving down) AND the centroid is below the
                        # center line, count the object
                        elif direction_x > 0 and centroid[0] > int(L_pos) and centroid[0] < int(L_pos)+10:
                            Exit += 1
                            to.counted = True
                            print(id, " Exiting")


            # store the trackable object in our dictionary
            trackableObjects[objectID] = to
            # draw both the ID of the object and the centroid of the
            # object on the output frame
            if axis == "H":
                cv2.line(frame, (0, int(L_pos)), (W, int(L_pos)), (255, 255, 0), 2)
            else:
                cv2.line(frame, (int(L_pos), 0), (int(L_pos), H), (255, 255, 0), 2)
            if centroid[0] > W-0.05*W or centroid[0] < 0.05*W or centroid[1] > H-0.1*H or centroid[1] < 0.1*H:
            # if not direction_x==0 or direction_y==0:
            # if not objectID:
                continue
            text = "ID {}".format(objectID)
            cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
        for h in pairs:
            cv2.line(frame, tuple(h[0]), tuple(h[1]), (0, 0, 255), 2)
    processedImg = frame.copy()
    return count,max_count

create = None
frameno = 0
filename = "videos/example_02.mp4"
yolo = "yolo-coco/"
opname = "output_videos/output_of_" + filename.split('/')[1][:-4] + '.mp4'
cap = cv2.VideoCapture(filename)

time1 = time.time()
print("Frame Size:", cap.read()[1].shape[0],"x",cap.read()[1].shape[1])
axis = input("Enter starting line axis(Horizontal or Vertical)(H/V):")
L_pos = input("Enter position:")
max_people=input("Max people allowed:")
Setup(yolo)
while True:
    frame_count += 1
    ret, frame = cap.read()
    if not ret:
        break
    current_img = frame.copy()
    current_img = imutils.resize(current_img)
    H, W, _ = current_img.shape
    frameno += 1

    if frameno % 2 == 0 or frameno == 1:
        count,max_count = ImageProcess(current_img)
        Frame = processedImg
    info = [
        ("Exit", Exit),
        ("Entry", Entry),
    ]
    for (i, (k, v)) in enumerate(info):
        text = "{}: {}".format(k, v)
        cv2.putText(Frame, text, (10, H - ((i * 20) + 60)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        if create is None:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            create = cv2.VideoWriter(opname, fourcc, 30, (Frame.shape[1], Frame.shape[0]), True)
    create.write(Frame)
    cv2.imshow("Frame",Frame)
    text = "Current misconduct count:"+str(count)
    cv2.putText(Frame,text,(10,H-30),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,255),2)
    text = "Max misconduct count:" + str(max_count)
    cv2.putText(Frame, text, (10, H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    if cv2.waitKey(1) & 0xFF == ord('q'):
       break
time2 = time.time()
print("Completed. Total Time Taken: {} minutes".format((time2-time1)/60))
print("Maximum Number of misconduct found:",max_count)
cap.release()
cv2.destroyAllWindows()