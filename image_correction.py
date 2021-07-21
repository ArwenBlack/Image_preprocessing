from PIL import Image
from skimage.filters import threshold_local
from cv2 import cv2
import imutils
from transform import four_point_transform
import pytesseract as tess
import re
import numpy as np
tess.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

image = cv2.imread("test_1.jpg", cv2.IMREAD_COLOR)
ratio = image.shape[0] / 500.0
orig = image.copy()
image = imutils.resize(image, height=500)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

cv2.imshow("Image", gray)
gray = cv2.GaussianBlur(gray, (5, 5), 0)

cv2.imshow("Image", image)
cv2.imshow("Gray", gray)

edged = cv2.Canny(gray, 75, 200)


cv2.imshow("Image", image)
cv2.imshow("Edged", edged)
cv2.waitKey(0)
cv2.destroyAllWindows()

cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5]

for c in cnts:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    if len(approx) == 4:
        screenCnt = approx
        break


cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 2)
cv2.imshow("Outline", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

warped = four_point_transform(orig, screenCnt.reshape(4, 2) * ratio)


cv2.imwrite("transformed.jpg", warped)

newdata=tess.image_to_osd("transformed.jpg")
angle = re.search('(?<=Orientation in degrees: )\d+', newdata).group(0)


rotated = imutils.rotate_bound(warped, float(angle))
result = tess.image_to_string(rotated)


result  = tess.image_to_data(rotated, output_type=tess.Output.DICT)
count = len(result)
conf = 0
for i in range(0, count):
    if(float(result["conf"][i]) != -1):
        conf += float(result["conf"][i])
confidence = conf / count
print(confidence)

if confidence <= 20:
    rotated = imutils.rotate_bound(rotated, 180)


cv2.imshow("Scanned", imutils.resize(rotated, height=500))


warped = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)

cv2.imshow("Scanned_bcla_and_white", imutils.resize(warped, height=500))
cv2.imshow("Original", imutils.resize(orig, height=500))

cv2.waitKey(0)




