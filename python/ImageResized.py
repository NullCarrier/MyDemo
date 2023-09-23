import cv2 as cv
import numpy as np
import sys


class Image_Tools:
    # class attributes:
    type_dict = {
        "RGB_FILE" : "RGB_FILE_TYPE",
        "BIN_FILE" : "BIN_FILE_TYPE"
    }


    def __init__(self, file_name, file_type, in_channel=3):
        self.fileName = file_name 
        self.type = file_type
        self.channel = in_channel
        self.image_data = None
    
    def Resize(self, dWidth, dHeight, resize_method=None):
        if self.type == Image_Tools.type_dict['RGB_FILE']:
            #read as bgr
            img = cv.imread(self.fileName, cv.IMREAD_COLOR) 
            if img is None:
                sys.exit("Could not read the image.")
            print("Original image size: ", img.shape) 
            #force using linear atm 
            imgRs = cv.resize(img, (dWidth, dHeight), interpolation = cv.INTER_AREA) 
            print("Resized image size: ", imgRs.shape)
            self.image_data = imgRs

        # May have bug on the binary resize
        elif self.type == Image_Tools.type_dict['BIN_FILE']:
            input_img = np.fromfile(self.fileName, dtype=np.uint8).reshape(dWidth, dHeight, channel)
            print("The resized image shape : ", input_img.shape)
    
            cv.imwrite(out_img_name + out_ext, input_img)
    
    def Dump_out(self, out_fileName, file_type=None): 

        out_img_name = out_fileName + '_' + "_3CHL" if self.channel == 3 else "_1CHL"   
        if file_type == Image_Tools.type_dict['RGB_FILE'] :
            out_ext = ".jpg"
            #write out as bgr format
            cv.imwrite( out_img_name + out_ext, self.image_data)
        elif file_type == Image_Tools.type_dict['BIN_FILE']:
            out_ext = ".bin"
            img_np = self.image_data

            if img_np is not None:
                img_np.tofile(out_img_name+out_ext)
            else:
                sys.exit("No data to dump out.")


    
            





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

    img_tool = Image_Tools(image, 'RGB_FILE_TYPE', 3)
    img_tool.Resize(1920, 1080)
    img_tool.Dump_out('img_1920x1080', 'BIN_FILE_TYPE')


    #Resize_3_CHL(image, width, height)
    #Resize_Binary(image, width, height, 3)
