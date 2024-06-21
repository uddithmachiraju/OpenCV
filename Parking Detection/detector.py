import cv2, csv, time
import pandas as pd 

class Spots:
    location = 0

def callback(foo):
    pass

columns = ['x1', 'y1', 'x2', 'y2']
roi_data = pd.read_csv('Data/rois.csv', names=columns)

cv2.namedWindow('parameters')
cv2.createTrackbar('Threshold1', 'parameters', 186, 700, callback)
cv2.createTrackbar('Threshold2', 'parameters', 122, 700, callback)
cv2.createTrackbar('Min pixels', 'parameters', 100, 1500, callback)
cv2.createTrackbar('Max pixels', 'parameters', 322, 1500, callback)

video = cv2.VideoCapture('Data/lot.mp4')

def drawRectangle(image, a, b, c, d, low_threshold, high_threshold, min_pix, max_pix):
    sub_image = image[b:b + d, a:a + c]
    edges = cv2.Canny(sub_image, low_threshold, high_threshold)
    pix = cv2.countNonZero(edges)
    if pix in range(min_pix, max_pix):
        cv2.rectangle(image, (a, b), (a + c, b + d), (0, 255, 0), 3)
        Spots.location += 1
    else:
        cv2.rectangle(image, (a, b), (a + c, b + d), (0, 0, 255), 3)

# Stream the video
while video.isOpened():
    Spots.location = 0
    ret, image = video.read()
    if not ret:
        break

    min_pix = cv2.getTrackbarPos('Min pixels', 'parameters')
    max_pix = cv2.getTrackbarPos('Max pixels', 'parameters')
    low_threshold = cv2.getTrackbarPos('Threshold1', 'parameters')
    high_threshold = cv2.getTrackbarPos('Threshold2', 'parameters')

    for i in range(roi_data.shape[0]): 
        drawRectangle(image, roi_data.iloc[i, 0], roi_data.iloc[i, 1], roi_data.iloc[i, 2], roi_data.iloc[i, 3],
                      low_threshold, high_threshold, min_pix, max_pix) 

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, 'Available Spots: ' + str(Spots.location),
                (10, 30), font, 1, (0, 255, 0), 3)
    cv2.imshow('Frame', image)

    canny = cv2.Canny(image, low_threshold, high_threshold)
    cv2.imshow('Canny', canny)

    time.sleep(0.5)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
