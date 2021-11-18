# Distance-Check---People-Counter

## Getting Started
- Download the Yolov3.weights from [here](https://drive.google.com/drive/u/0/folders/1PfDMFh95obu40N-8tLk-EAF7x847EnnM) .
  - Load up the Videos you wanna use in `./videos`
  - Go to the `SocialDistancingDetector.py`  change the filename variable to `videos/[filename.mp4]`
  - If you wanna test on live feed change the filename `0` (for native webcam) or the port you've connected the webcam to...
+ *After you've successfully made the required changes*
+ *Head to the terminal/cmd line by right-clicking on the desired path* 

![image](https://user-images.githubusercontent.com/56502015/142374819-2605b52f-d699-46a0-8605-69519125d794.png) 
+ *Type `python SocialDistancingDetector.py` or `python3 SocialDistancingDetector.py`(if on linux)*
+ The output videos are stored in the `./output_videos`.


Here is sample video output:<br><br>

https://user-images.githubusercontent.com/56502015/142375853-bea54c5f-2af1-437f-9e5d-85fd004fd43a.mp4

## Object detection:
- We will be using YOLOv3, trained on COCO dataset for object detection.
- In general, single-stage detectors like YOLO tend to be less accurate than two-stage detectors (R-CNN) but are significantly faster.
- YOLO treats object detection as a regression problem, taking a given input image and simultaneously learning bounding box coordinates and corresponding class label probabilities.
- It is used to return the person prediction probability, bounding box coordinates for the detection, and the centroid of the person.

## Distance calculation:
- NMS (Non-maxima suppression) is also used to reduce overlapping bounding boxes to only a single bounding box, thus representing the true detection of the object. Having overlapping boxes is not exactly practical and ideal, especially if we need to count the number of objects in an image.
- Euclidean distance is then computed between all pairs of the returned centroids. Simply, a centroid is the center of a bounding box.
- Based on these pairwise distances, we check to see if any two people are less than/close to 'N' pixels apart.

## References
- YOLOv3 paper: https://arxiv.org/pdf/1804.02767.pdf
- YOLO original paper: https://arxiv.org/abs/1506.02640
- YOLO TensorFlow implementation (darkflow): https://github.com/thtrieu/darkflow
