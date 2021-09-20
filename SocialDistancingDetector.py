import numpy as np
import cv2
import imutils
import os
import time
import random

def Check(a,  b, image):
    dist = ((a[0] - b[0]) ** 2 + 550 / ((a[1] + b[1]) / 2) * (a[1] - b[1]) ** 2) ** 0.5
    text = "Distance:"+str(dist)
    cv2.putText(image , text, (10, H - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
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

def ImageProcess(image):
    count = 0
    Entry = 0
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
                close = Check(center[i], center[j], image)

                if close:
                    pairs.append([center[i], center[j]])
                    count += 1
                    status[i] = True
                    status[j] = True
        index = 0
        for i in flat_box:
            (x, y) = (outline[i][0], outline[i][1])
            (w, h) = (outline[i][2], outline[i][3])
            if status[index] == True:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 150), 2)
                cv2.circle(frame, (x + w // 2, y + h // 2), 2, (0, 255, 0), -1)
            elif status[index] == False:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.circle(frame, (x + w // 2, y + h // 2), 2, (0, 255, 0), -1)
            text = "ID {}".format(index)
            cv2.putText(frame, text, (x + w // 2 - 10, y + h // 2 - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
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