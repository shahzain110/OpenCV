import cap as cap
import cv2
import numpy as np
from matplotlib import pyplot as plt

# displaying img
def showImage():
    img = cv2.imread('watch.jpg',cv2.IMREAD_GRAYSCALE)
    cv2.imshow('image',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# drawing lines
def drawLines(img):
    plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.plot([200,300,400],[100,200,300],'c', linewidth=5)
    plt.show()

# load video from cam
def loadVideoFromCam():
    cap = cv2.VideoCapture(1)

    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        cv2.imshow('frame', gray)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# draw and write on frames
def drawandWrite():
    img = cv2.imread('watch.jpg', cv2.IMREAD_COLOR)
    cv2.line(img, (0, 0), (200, 300), (255, 255, 255), 50)
    cv2.rectangle(img, (500, 250), (1000, 500), (0, 0, 255), 15)
    cv2.circle(img, (447, 63), 63, (0, 255, 0), -1)
    pts = np.array([[100, 50], [200, 300], [700, 200], [500, 100]], np.int32)
    pts = pts.reshape((-1, 1, 2))
    cv2.polylines(img, [pts], True, (0, 255, 255), 3)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, 'OpenCV Tuts!', (10, 500), font, 6, (200, 255, 155), 13, cv2.LINE_AA)
    cv2.imshow('image', img)
    cv2.waitKey(0)

    cv2.destroyAllWindows()

# extracting ROI from frames
def extractingROI():
    img = cv2.imread('watch.jpg', cv2.IMREAD_COLOR)
    px = img[55, 55]
    img[55, 55] = [255, 255, 255]
    px = img[55, 55]
    print(px)
    px = img[100:150, 100:150]
    print(px)
    img[100:150, 100:150] = [255, 255, 255]
    print(img.shape)
    print(img.size)
    print(img.dtype)
    watch_face = img[37:111, 107:194]
    img[0:74, 0:87] = watch_face

    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# performing arthimetic operations
def imageOperations():
    img1 = cv2.imread('3D-Matplotlib.png')
    img2 = cv2.imread('mainsvmimage.png')

    weighted = cv2.addWeighted(img1, 0.6, img2, 0.4, 0)
    cv2.imshow('weighted', weighted)

    # add = img1 + img2
    add = cv2.add(img1, img2)

    cv2.imshow('add', add)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# placing logo
def logoPlace():
    img1 = cv2.imread('3D-Matplotlib.png')
    img2 = cv2.imread('mainlogo.png')

    # I want to put logo on top-left corner, So I create a ROI
    rows, cols, channels = img2.shape
    roi = img1[0:rows, 0:cols]

    # Now create a mask of logo and create its inverse mask
    img2gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # add a threshold
    ret, mask = cv2.threshold(img2gray, 220, 255, cv2.THRESH_BINARY_INV)

    mask_inv = cv2.bitwise_not(mask)

    # Now black-out the area of logo in ROI
    img1_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)

    # Take only region of logo from logo image.
    img2_fg = cv2.bitwise_and(img2, img2, mask=mask)

    dst = cv2.add(img1_bg, img2_fg)
    img1[0:rows, 0:cols] = dst

    cv2.imshow('res', img1)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# multiple functions for thresholding
def thresholding():
    img = cv2.imread('bookpage.jpg')
    retval, threshold = cv2.threshold(img, 12, 255, cv2.THRESH_BINARY)
    cv2.imshow('original', img)
    cv2.imshow('threshold', threshold)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    grayscaled = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    th = cv2.adaptiveThreshold(grayscaled, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 1)
    cv2.imshow('original', img)
    cv2.imshow('Adaptive threshold', th)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# selecting a colors from image (bitwise operation)
def colorRangeSelection():
    cap = cv2.VideoCapture(0)

    while (1):
        _, frame = cap.read()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lower_red = np.array([30, 150, 50])
        upper_red = np.array([255, 255, 180])

        mask = cv2.inRange(hsv, lower_red, upper_red)
        res = cv2.bitwise_and(frame, frame, mask=mask)

        cv2.imshow('frame', frame)
        cv2.imshow('mask', mask)
        cv2.imshow('res', res)

        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            break

    cv2.destroyAllWindows()
    cap.release()

# blurring and smoothing a mask for good selections
def blur_smoothingImg():
    cap = cv2.VideoCapture(0)

    while (1):
        _, frame = cap.read()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lower_red = np.array([30, 150, 50])
        upper_red = np.array([255, 255, 180])

        mask = cv2.inRange(hsv, lower_red, upper_red)
        res = cv2.bitwise_and(frame, frame, mask=mask)
        kernel = np.ones((15, 15), np.float32) / 225
        smoothed = cv2.filter2D(res, -1, kernel)

        blur = cv2.GaussianBlur(res, (15, 15), 0)
        cv2.imshow('Gaussian Blurring', blur)

        median = cv2.medianBlur(res, 15)
        cv2.imshow('Median Blur', median)

        bilateral = cv2.bilateralFilter(res, 15, 75, 75)
        cv2.imshow('bilateral Blur', bilateral)

        cv2.imshow('Original', frame)
        cv2.imshow('Averaging', smoothed)

        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            break

    cv2.destroyAllWindows()
    cap.release()

# Erosion and Dilation. used for reduce noice in edges
def morphologicalTrans():
    cap = cv2.VideoCapture(0)

    while (1):

        _, frame = cap.read()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lower_red = np.array([30, 150, 50])
        upper_red = np.array([255, 255, 180])

        mask = cv2.inRange(hsv, lower_red, upper_red)
        res = cv2.bitwise_and(frame, frame, mask=mask)

        kernel = np.ones((5, 5), np.uint8)
        erosion = cv2.erode(mask, kernel, iterations=1)
        dilation = cv2.dilate(mask, kernel, iterations=1)

        opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        closing = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        cv2.imshow('Original', frame)
        cv2.imshow('Mask', mask)

        cv2.imshow('Erosion', erosion)
        cv2.imshow('Dilation', dilation)

        cv2.imshow('Opening', opening)
        cv2.imshow('Closing', closing)

        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            break

    cv2.destroyAllWindows()
    cap.release()

# CANNY EDGE DETECTION and GRADIENTS
def edgeDetection_Gradients():
    cap = cv2.VideoCapture(1)

    while (1):

        # Take each frame
        _, frame = cap.read()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lower_red = np.array([30, 150, 50])
        upper_red = np.array([255, 255, 180])

        mask = cv2.inRange(hsv, lower_red, upper_red)
        res = cv2.bitwise_and(frame, frame, mask=mask)

        # code 1
        laplacian = cv2.Laplacian(frame, cv2.CV_64F)
        sobelx = cv2.Sobel(frame, cv2.CV_64F, 1, 0, ksize=5)
        sobely = cv2.Sobel(frame, cv2.CV_64F, 0, 1, ksize=5)

        cv2.imshow('Original', frame)
        cv2.imshow('Mask', mask)
        cv2.imshow('laplacian', laplacian)
        cv2.imshow('sobelx', sobelx)
        cv2.imshow('sobely', sobely)

        # code 2
        mask = cv2.inRange(hsv, lower_red, upper_red)
        res = cv2.bitwise_and(frame, frame, mask=mask)

        cv2.imshow('Original', frame)
        edges = cv2.Canny(frame, 100, 200)
        cv2.imshow('Edges', edges)

        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            break

    cv2.destroyAllWindows()
    cap.release()

# template matching python
def templateMatch():
    # templateMatch()
    img_rgb = cv2.imread('opencv-template-matching-python-tutorial.jpg')
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

    template = cv2.imread('opencv-template-for-matching.jpg', 0)
    w, h = template.shape[::-1]

    res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)
    threshold = 0.8
    loc = np.where(res >= threshold)

    for pt in zip(*loc[::-1]):
        cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0, 255, 255), 2)

    cv2.imshow('Detected', img_rgb)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# extracting foreground
def foregroundExtract():
    from matplotlib import pyplot as plt

    img = cv2.imread('opencv-python-foreground-extraction-tutorial.jpg')
    mask = np.zeros(img.shape[:2], np.uint8)

    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    rect = (161, 79, 150, 150)

    cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    img = img * mask2[:, :, np.newaxis]

    plt.imshow(img)
    plt.colorbar()
    plt.show()

# will detect corners
def cornerDetection():
    img = cv2.imread('opencv-corner-detection-sample.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)

    corners = cv2.goodFeaturesToTrack(gray, 100, 0.01, 10)
    corners = np.int0(corners)

    for corner in corners:
        x, y = corner.ravel()
        cv2.circle(img, (x, y), 3, 255, -1)

    cv2.imshow('Corner', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# homograph feature matching
def featureMatch():
    img1 = cv2.imread('opencv-feature-matching-template.jpg', 0)
    img2 = cv2.imread('opencv-feature-matching-image.jpg', 0)
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:10], None, flags=2)
    plt.imshow(img3)
    plt.show()


def backgroundRemoval():
    cap = cv2.VideoCapture('people-walking.mp4')
    fgbg = cv2.createBackgroundSubtractorMOG2()

    while (1):
        ret, frame = cap.read()

        fgmask = fgbg.apply(frame)

        cv2.imshow('fgmask', frame)
        cv2.imshow('frame', fgmask)

        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


# HAAR CASCADE FACE EYE DETECTION
def objectDetection():
    import numpy as np
    import cv2

    # multiple cascades: https://github.com/Itseez/opencv/tree/master/data/haarcascades

    # https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    # https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_eye.xml
    eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

    cap = cv2.VideoCapture(0)

    while 1:
        ret, img = cap.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = img[y:y + h, x:x + w]

            eyes = eye_cascade.detectMultiScale(roi_gray)
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

        cv2.imshow('img', img)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

def main():
    pass
# add a method here tp run


if __name__ == "__main__":
    main()