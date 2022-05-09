import cv2

CAM_WIDTH = 1280
CAM_HEIGHT = 720
cap = cv2.VideoCapture(0)
cap.set(3, CAM_WIDTH)
cap.set(4, CAM_HEIGHT)

face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

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
