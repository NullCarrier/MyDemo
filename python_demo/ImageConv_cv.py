import cv2 
import numpy as np
import sys


    
def CV_Conv(imgFile):
    img= cv2.imread(imgFile)
    #cv2.imshow('Original', img)

    #Identity kernel
    kernel1 = np.array([[0,0,0], [0, 1,0], [0,0,0]])
    im1 = cv2.filter2D(img, -1, kernel1)
    cv2.imwrite('Identity_kernel.png', im1)

    #shapening kernel: 3x3
    kernel2 = np.array([[-1,-1,-1], [-1, 9,-1], [-1,-1,-1]])
    im2 = cv2.filter2D(img, -1, kernel2)
    cv2.imwrite('Sharpening_kernel.png', im2)

    #blurring kernel
    kernel3 = np.array([[0.02040816, 0.02040816, 0.02040816, 0.02040816, 0.02040816,
        0.02040816, 0.02040816],
       [0.02040816, 0.02040816, 0.02040816, 0.02040816, 0.02040816,
        0.02040816, 0.02040816],
       [0.02040816, 0.02040816, 0.02040816, 0.02040816, 0.02040816,
        0.02040816, 0.02040816],
       [0.02040816, 0.02040816, 0.02040816, 0.02040816, 0.02040816,
        0.02040816, 0.02040816],
       [0.02040816, 0.02040816, 0.02040816, 0.02040816, 0.02040816,
        0.02040816, 0.02040816],
       [0.02040816, 0.02040816, 0.02040816, 0.02040816, 0.02040816,
        0.02040816, 0.02040816],
       [0.02040816, 0.02040816, 0.02040816, 0.02040816, 0.02040816,
        0.02040816, 0.02040816]])
    im3 = cv2.filter2D(img, -1, kernel3)
    cv2.imwrite('Blurring_kernel.png', im3)



#############################Main function
if __name__ == "__main__":

    #Total Argument counts
    argc = len(sys.argv) 
    if argc < 2:
        sys.exit("Usage: fileName ")

    #Load input argu
    image = sys.argv[1]

    CV_Conv(image)
