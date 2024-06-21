import cv2, csv, time

video = cv2.VideoCapture('Data/lot.mp4') 
source, image = video.read()
cv2.imwrite("frame.jpg", image)
video.release()
cv2.destroyAllWindows()
image = cv2.imread("frame.jpg")
 
r = cv2.selectROI('Selector', image, showCrosshair=False, fromCenter=False)
rlist = list(r)

with open('Data/rois1.csv', 'a', newline='') as outf:
    csvw = csv.writer(outf)
    csvw.writerow(rlist)