import sys

from PIL import Image
from skimage.filters import threshold_local
from cv2 import cv2
import imutils
import pytesseract as tess
import re
import numpy as np
tess.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'

def resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    dim = None
    (h,w)  = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        r = height/float(h)
        dim = (int(w*r), height)
    else:
        r = width/float(w)
        dim = (width, int(h*r))
    return cv2.resize(image, dim, interpolation=inter)


def grab_contours(cnts):
    if len(cnts) == 2:
        cnts = cnts[0]
    elif len(cnts) == 3:
        cnts = cnts[1]
    else:
        raise Exception(("Nope"))
    return cnts


def order_points(points):
    rectangle = np.zeros((4, 2), dtype="float32")
    s = points.sum(axis=1)
    rectangle [0] = points[np.argmin(s)]
    rectangle [2] = points[np.argmax(s)]
    diff = np.diff(points, axis=1)
    rectangle [1] = points[np.argmin(diff)]
    rectangle [3] = points[np.argmax(diff)]
    return rectangle

def four_point_transform(image, points):
    rect = order_points(points)
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped


def rotate_full(image, angle):
    (h,w) = image.shape[:2]
    (cX, cY) = (w/2, h/2)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0,0])
    sin = np.abs(M[0,1])
    nW = int((h*sin)+(w*cos))
    nH = int((h*cos)+(w*sin))
    M[0,2] += (nW/2) - cX
    M[1,2] += (nH/2) - cY
    return cv2.warpAffine(image, M, (nW, nH))

im_path = sys.argv[1]

image = cv2.imread(im_path, cv2.IMREAD_COLOR)
ratio = image.shape[0] / 500.0
orig = image.copy()
image = imutils.resize(image, height=500)
#cv2.imshow("Image", image)
image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#cv2.imshow("Gray", image_gray)
image_gray= cv2.GaussianBlur(image_gray, (5, 5), 0)
#cv2.imshow("Gray with blur", image_gray)
edged = cv2.Canny(image_gray, 75, 200)
#cv2.imshow("Edged", edged)

#cv2.waitKey(0)

contours = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(contours)
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

for c in contours:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    if len(approx) == 4:
        imageCnt = approx
        break

try:
    cv2.drawContours(image, [imageCnt], -1, (0,255,0),2)
    #cv2.imshow("Contour", image)
    pretty = four_point_transform(orig, imageCnt.reshape(4, 2) * ratio)
except:
    image1 = cv2.copyMakeBorder(image, 5,5,5,5, cv2.BORDER_CONSTANT, value = [0,255,0])
    #cv2.imshow("Contour", image1)
    pretty = image

cv2.imwrite("transformed.jpg", pretty)
#cv2.waitKey(0)

try:
    newdata=tess.image_to_osd("transformed.jpg")
    print(newdata)
    angle = re.search('(?<=Orientation in degrees: )\d+', newdata).group(0)
    print(angle)
    rotated = imutils.rotate_bound(pretty, float(angle))
except:
    rotated = pretty



result  = tess.image_to_data(rotated, output_type=tess.Output.DICT)
count = len(result["conf"])
conf = 0
for i in range(0, count):
    conf += float(result["conf"][i])
confidence = conf / count
confidence_new = 100
pom = 1
rotated_new = rotated
while pom <= 4:
    rotated_new = rotate_full(rotated_new, 90)
    result = tess.image_to_data(rotated_new, output_type=tess.Output.DICT)
    count = len(result["conf"])
    conf = 0
    for i in range(0,count):
        conf += float(result["conf"][i])
    confidence_new = conf/count
    print([confidence, confidence_new])
    if confidence_new>confidence:
        rotated = rotated_new
        confidence = confidence_new
    pom+=1

final = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
#cv2.imshow("Scanned", final)
#cv2.waitKey(0)
cv2.imwrite(im_path[:-3] + "_corrected.jpg", final)

# final_edit = final
# #cienie
# rgb_planes = cv2.split(final_edit)
# result_planes = []
# result_norm_planes = []
# warped = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
#
# cv2.imshow("Scanned_bcla_and_white", imutils.resize(warped, height=500))
# cv2.imshow("Original", imutils.resize(orig, height=500))
#
# cv2.waitKey(0)




