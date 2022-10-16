import cv2 as cv
import numpy as np
import sys


def Resize_3_CHL(image, dWidth, dHeight):
    #Read as BGR
    img = cv.imread(image, cv.IMREAD_COLOR) 
    if img is None:
        sys.exit("Could not read the image.")
    print("Original image size: ", img.shape) 

    imgRs = cv.resize(img, (dWidth, dHeight), interpolation = cv.INTER_AREA) 
    print("Resized image size: ", imgRs.shape)
    
    cv.imwrite("Test_1920x1080.jpg", imgRs)
    #imgRs.tofile("Test_1920x1080.bin")


def Resize_Binary(file_name, dWidth, dHeight, channel):
    input_img = np.fromfile(file_name, dtype=np.uint8).reshape(dWidth, dHeight, channel)

    print("The resized image shape : ", input_img.shape)

    out_img_name = "img_" + str(dWidth) + "x" + str(dHeight) + "_3CHL" if channel == 3 else "_1CHL"   
    out_ext = ".png"
    
    cv.imwrite(out_img_name+out_ext, input_img)
    



#############################Main function
if __name__ == "__main__":

    #Total Argument counts
    argc = len(sys.argv) 
    if argc < 4:
        sys.exit("Usage: fileName width height")

    #Load input argu
    image = sys.argv[1]
    width = int(sys.argv[2])
    height = int(sys.argv[3])

    #Resize_3_CHL(image, width, height)
    Resize_Binary(image, width, height, 3)
