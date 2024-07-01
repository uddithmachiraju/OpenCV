import cv2 
import numpy as np 

# Morphology 
kernel = np.ones((5, 5), dtype = np.uint8) 
original_image = cv2.imread('Data/test_2.jpg')
image = cv2.morphologyEx(original_image, cv2.MORPH_CLOSE, kernel = kernel) 

# Create an initial mask
mask = np.zeros(image.shape[:2], np.uint8)

# Create temporary arrays used by the algorithm
bgdModel = np.zeros((1, 65), np.float64)
fgdModel = np.zeros((1, 65), np.float64)

# Define the rectangle around the object
rect = (50, 50, image.shape[1] - 80, image.shape[0] - 80)

# Apply GrabCut algorithm with the rectangle
cv2.grabCut(image, mask, rect, bgdModel, fgdModel, 10, cv2.GC_INIT_WITH_RECT)

mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

# Multiply the original image with the new mask to remove the background
image = image * mask2[:, :, np.newaxis]

# Convert to grayscale and blur
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (11, 11), 0)

# Edge detection
canny = cv2.Canny(gray, 0, 200)
dilated = cv2.dilate(canny, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

# Find contours
contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Loop over the contours to find the document
for c in contours:
    # Approximate the contour
    epsilon = 0.02 * cv2.arcLength(c, True)
    corners = cv2.approxPolyDP(c, epsilon, True)
    # If our approximated contour has four points, we assume it's the document
    if len(corners) == 4:
        break

# Draw the contours and corners on the image
con = np.zeros_like(image) 
cv2.drawContours(con, [c], -1, (0, 255, 255), 3)
cv2.drawContours(con, [corners], -1, (0, 255, 0), 10)

# Sort corners and label them
corners = sorted(np.concatenate(corners).tolist())
for index, corner in enumerate(corners):
    character = chr(65 + index)
    cv2.putText(con, character, tuple(corner), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3, cv2.LINE_AA)

def order_points(pts):
    rect = np.zeros((4, 2), dtype='float32')
    pts = np.array(pts)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect.astype('int').tolist()

# Order the corners
ordered_corners = order_points(corners)

(tl, tr, br, bl) = ordered_corners

# Finding the maximum width
widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
maxWidth = max(int(widthA), int(widthB))

# Finding the maximum height
heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
maxHeight = max(int(heightA), int(heightB))

# Final destination coordinates
destination_corners = np.array([
    [0, 0],
    [maxWidth - 1, 0],
    [maxWidth - 1, maxHeight - 1],
    [0, maxHeight - 1]], dtype="float32")

# Perspective transform
M = cv2.getPerspectiveTransform(np.array(ordered_corners, dtype="float32"), destination_corners)
warped = cv2.warpPerspective(original_image, M, (maxWidth, maxHeight)) 

image = cv2.resize(warped, (int(750 * 0.7), int(1000 * 0.7)))
cv2.imshow("Document Scanner", image) 
cv2.waitKey(0)
cv2.destroyAllWindows() 