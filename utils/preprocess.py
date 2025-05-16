import cv2
import numpy as np

def mobile_preprocess(image):
    img = remove_border(image)
    img = edge_enhancement(img)
    img = denoise(img)
    img = apply_sharpening(img)
    return img 

def remove_border(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    c = max(contours, key=cv2.contourArea)
    (x, y), r = cv2.minEnclosingCircle(c)
    mask = np.zeros_like(gray)
    cv2.circle(mask, (int(x), int(y)), int(r), 255, -1)
    img = cv2.bitwise_and(img, img, mask=mask)
    x, y, w, h = cv2.boundingRect(c)
    img = img[y:y+h, x:x+w]
    return img

def apply_clahe(img,cliplimit=36.74408702597458,tilegridsize=31):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=cliplimit, tileGridSize=tilegridsize)
    cl = clahe.apply(l)
    merged = cv2.merge((cl, a, b))
    processed = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
    return processed

def apply_sharpening(image):
    sharpening_kernel = np.array([
        [0, -1, 0],
        [-1, 5, -1],
        [0, -1, 0]
    ])
    sharpened = cv2.filter2D(image, -1, sharpening_kernel)
    return sharpened

def denoise(img):
    return cv2.fastNlMeansDenoisingColored(img, None, 1, 1, 7, 21)

def edge_enhancement(image):
    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    return cv2.convertScaleAbs(image + laplacian)
