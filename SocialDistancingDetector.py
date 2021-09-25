import numpy as np
import cv2
import imutils
import os
import time
from centroidtracker import CentroidTracker
from trackableobject import TrackableObject
dist=0

ct = CentroidTracker(maxDisappeared=100, maxDistance=100)
trackers = []
trackableObjects = {}
Entry = 0
Exit = 0

def Check(a,  b, image):
    dist = ((a[0] - b[0]) ** 2 + 550 / ((a[1] + b[1]) / 2) * (a[1] - b[1]) ** 2) ** 0.5
    # text = "Distance:{}".format(int(dist))
    # cv2.putText(image, text, (10, H - 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    calibration = (a[1] + b[1]) / 2       
    if 0 < dist < 0.25 * calibration:
        return True
    else:
        return False

def Setup(yolo):
    global net, ln, LABELS
    weights = os.path.sep.join([yolo, "yolov3.weights"])
    config = os.path.sep.join([yolo, "yolov3.cfg"])
    labelsPath = os.path.sep.join([yolo, "coco.names"])
    LABELS = open(labelsPath).read().strip().split("\n")  
    net = cv2.dnn.readNetFromDarknet(config, weights)
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]

def ImageProcess(image,Entry,Exit):
    count = 0
    rects = []
    Entry = 25
    Exit = 0
    global processedImg
    (H, W) = (None, None)
    frame = image.copy()
    if W is None or H is None:
        (H, W) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    starttime = time.time()
    layerOutputs = net.forward(ln)
    stoptime = time.time()
    print("Video is Getting Processed at {:.4f} seconds per frame".format((stoptime-starttime)))
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

        for i in range(len(center)):

            for j in range(len(center)):
                close = Check(center[i], center[j], frame)

                if close:
                    pairs.append([center[i], center[j]])
                    count += 1
                    status[i] = True
                    status[j] = True
        index = 0
        for i in flat_box:
            (x, y) = (outline[i][0], outline[i][1])
            (w, h) = (outline[i][2], outline[i][3])
            # cv2.circle(frame, (x + w // 2, y + h // 2), 2, (0, 255, 0), -1)
            if status[index] == True:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 150), 2)
            elif status[index] == False:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            rects.append((x, y, x+w, y+h))
            # text = "ID {}".format(index)
            # cv2.putText(frame, text, (x + w // 2 - 10, y + h // 2 - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
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
                            if direction_y < 0 and centroid[1] < int(L_pos):
                                Entry += 1
                                to.counted = True
                                print(id, " Entering")
                            # if the direction is positive (indicating the object
                            # is moving down) AND the centroid is below the
                            # center line, count the object
                            elif direction_y > 0 and centroid[1] > int(L_pos):
                                Exit += 1
                                to.counted = True
                                print(id, " Exiting")
                        if axis == "V":
                            if direction_x < 0 and centroid[0] < int(L_pos):
                                Entry += 1
                                to.counted = True
                                print(id, " Entering")
                            # if the direction is positive (indicating the object
                            # is moving down) AND the centroid is below the
                            # center line, count the object
                            elif direction_x and centroid[0] > int(L_pos):
                                Exit += 1
                                to.counted = True
                                print(id, " Exiting")
                # store the trackable object in our dictionary
                trackableObjects[objectID] = to
                # totalExit = Exit
                # totalEntry = Entry
                # totalExit += Exit
                # totalEntry += Entry
                # draw both the ID of the object and the centroid of the
                # object on the output frame
                text = "ID {}".format(objectID)
                cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
            index += 1
            if axis == "H":
                cv2.line(frame, (0, int(L_pos)), (W, int(L_pos)), (255, 255, 0), 2)
            else:
                cv2.line(frame, (int(L_pos), 0), (int(L_pos), H), (255, 255, 0), 2)
            info = [
                ("Exit", Exit),
                ("Entry", Entry),
            ]
            for (i, (k, v)) in enumerate(info):
                text = "{}: {}".format(k, v)
                cv2.putText(frame, text, (10, H - ((i * 20) + 40)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        for h in pairs:
            cv2.line(frame, tuple(h[0]), tuple(h[1]), (0, 0, 255), 2)
    processedImg = frame.copy()
    return count

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
while(True):
    ret, frame = cap.read()
    if not ret:
        break
    current_img = frame.copy()
    current_img = imutils.resize(current_img)
    H,W,_ = current_img.shape
    frameno += 1

    if(frameno%2 == 0 or frameno == 1):
        Setup(yolo)
        count = ImageProcess(current_img)
        Frame = processedImg

        if create is None:
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            create = cv2.VideoWriter(opname, fourcc, 30, (Frame.shape[1], Frame.shape[0]), True)
    create.write(Frame)
    cv2.imshow("Frame",Frame)
    text = "Total misconduct count:"+str(count)
    cv2.putText(Frame,text,(10,H-10),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,255),2)
    if cv2.waitKey(1) & 0xFF == ord('q'):
       break
time2 = time.time()
print("Completed. Total Time Taken: {} minutes".format((time2-time1)/60))
print("Number of misconduct found:",count)
cap.release()
cv2.destroyAllWindows()