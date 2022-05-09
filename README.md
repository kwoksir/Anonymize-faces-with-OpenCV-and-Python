# Anonymize face with OpenCV
<img src="https://user-images.githubusercontent.com/61585411/167336213-74f00f1f-ccda-4c37-acd0-18e2dc4c9a87.jpg" width=800>

## Procedures
1. Import the library.
2. Setting up a webcam.
3. Creating face detector
4. Defining a mosaic function to pixelate the detected face region
5. Displaying the output

## Step 1: Import the library
```python
import cv2
```
## Step 2: Setting up a webcam (Windows)
```python
CAM_WIDTH = 1280
CAM_HEIGHT = 720
cap = cv2.VideoCapture()
cap.open(0, cv2.CAP_DSHOW)
cap.set(3, CAM_WIDTH)
cap.set(4, CAM_HEIGHT)
```
It is quicker to get web cam live in Windows environment by adding cv2.CAP_DSHOW attribute.
## Step 2: Setting up a webcam (Windows/Linux/Mac)
```python
CAM_WIDTH = 1280
CAM_HEIGHT = 720
cap = cv2.VideoCapture(0)
cap.set(3, CAM_WIDTH)
cap.set(4, CAM_HEIGHT)
```
## Step 3: Creating face detector
```python
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
```
## Step 4: Defining a mosaic function to pixelate the detected face region
```python
def mosaic(img, rect, size):
    (x1, y1, x2, y2) = rect
    w = x2 - x1
    h = y2 - y1
    i_rect = img[y1:y2, x1:x2]
    i_small = cv2.resize(i_rect, (size, size))
    i_mos = cv2.resize(i_small, (w,h), interpolation=cv2.INTER_AREA)
    img2 = img.copy()
    img2[y1:y2, x1:x2] = i_mos
    return img2
```
## Step 5: Displaying the output
Performing face detection, then extract the face region of interest. Apply the anonymize function to pixelate the region and then store the pixelated face in the original image.
```python
while cap.isOpened():
    success, img = cap.read()
    if success:
        faces = face_detector.detectMultiScale(img, scaleFactor=1.2, minNeighbors=5, minSize=(50,50))
    for (x,y,w,h) in faces:
        img = mosaic(img, (x,y, x+w, y+h), 10)
        cv2.rectangle(img,(x,y), (x+w,y+h), (0,255,0),2)
    cv2.imshow('demo',img)
    if cv2.waitKey(1) == ord('q') or cv2.waitKey(1) == 27:
        break
else:
    print('Fail to open')
cv2.destroyAllWindows()
cap.release()
```
## References
- [Blur and anonymize faces with OpenCV and Python (Pyimagesearch)](https://pyimagesearch.com/2020/04/06/blur-and-anonymize-faces-with-opencv-and-python/)
- [Blur and anonymize faces with OpenCV and Python (Geeksforgeeks)](https://www.geeksforgeeks.org/blur-and-anonymize-faces-with-opencv-and-python/)

