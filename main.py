import sys
import cv2
import numpy as np

def main():
    sys.argv.pop(0)
    for image_filename in sys.argv:
        analyzeImage(image_filename)

def analyzeImage(image_filename):
    raw_image = cv2.imread(image_filename)
    grayscale_image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2GRAY)
    constrasted_image = cv2.equalizeHist(grayscale_image)
    displayImage(constrasted_image, "Constrasted Image")

    lower = np.array([230], dtype = "uint16")
    upper = np.array([255], dtype = "uint16")
    constrasted_mask = cv2.inRange(constrasted_image, lower, upper)
    
    kernel = np.ones((5,5),np.uint8)
    constrasted_mask = cv2.morphologyEx(constrasted_mask, cv2.MORPH_CLOSE, kernel)
    displayImage(constrasted_mask, "Constrasted Mask")

    masked_image = cv2.bitwise_and(raw_image, raw_image, mask=constrasted_mask)
    # displayImage(masked_image, "Masked Image")

def displayImage(image, title="Untitled"):
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()