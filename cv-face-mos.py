import cv2

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

CAM_WIDTH = 1920
CAM_HEIGHT = 1080
face_detector = cv2.CascadeClassifier('face.xml')
capture = cv2.VideoCapture(0)
capture.set(3, CAM_WIDTH)
capture.set(4, CAM_HEIGHT)

while capture.isOpened():
    success, img = capture.read()
    if success:
        faces = face_detector.detectMultiScale(img, scaleFactor=1.2, minNeighbors=5, minSize=(50,50))
    for (x,y,w,h) in faces:
        img = mosaic(img, (x,y, x+w, y+h), 20)
        cv2.rectangle(img,(x,y), (x+w,y+h), (0,255,0),2)
        cv2.putText(img,"face",(x,y-5),cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0,255,255), 1)

    cv2.imshow('Frame1',img)
    if cv2.waitKey(1) == ord('q') or cv2.waitKey(1) == 27:
        cv2.destroyAllWindows()
        capture.release()
        break
else:
    print('Fail to open')
