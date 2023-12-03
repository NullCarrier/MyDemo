import cv2 as cv
import numpy as np
import sys


def Compare_data(golden_data, in_data):
    from scipy.spatial import distance

    print("----------------------------------")

    golden_data_fp = golden_data.astype(np.float32).ravel()
    in_data_fp = in_data.astype(np.float32).ravel()

    if golden_data_fp.shape != in_data_fp.shape:
        print(f"E: diff in shape: {golden_data_fp.shape} vs {in_data_fp.shape}")  
        return False

    cosine_dst = 1 - distance.cosine(in_data_fp, golden_data_fp)
    euc = distance.euclidean(in_data_fp, golden_data_fp)

    print(f"cosine = {cosine_dst:.6f}, euclidean = {euc:.4f}")
    print("----------------------------------")

    return True






#############################Main function
if __name__ == "__main__":

    #Total Argument counts
    # argc = len(sys.argv) 
    # if argc < 4:
    #     sys.exit("Usage: fileName width height")

    # #Load input argu
    # image = sys.argv[1]
    # width = int(sys.argv[2])
    # height = int(sys.argv[3])

    # data_0 = np.array([1000,100])
    # data_1 = np.array([500,50])

    data_0 = np.array([[1, 0, 0], [0, 1, 0]])
    data_1 = np.array([[1, 0, 1], [0, 1, 0]])
   
    Compare_data(data_0, data_1)
