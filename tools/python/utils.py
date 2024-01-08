import cv2 as cv
import numpy as np
import sys

DataType_lut_str2npy = {
    'float' : np.float32,
    'float16' : np.float16,
    'int8': np.int8,
    'uint8': np.uint8,
    'int32': np.int32
    # to do, complex 
}


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


'''
load data in file from given path into numpy array
'''
def Load_files(input_file_path, data_type='float'):
    from pathlib import Path
    # extract file name
    path_obj = Path(input_file_path)
    file_name = path_obj.name 
    file_ext = path_obj.suffix 

    print(f"the file name: {file_name} with ext: {file_ext}")
    data_np = None
    try:
        if file_ext == ".npy":
            data_np = np.load(input_file_path)
        elif file_ext == ".bin": 
            # using default type float
            type_np = DataType_lut_str2npy[data_type]
            print("The data type: ",type_np)
            data_np = np.fromfile(input_file_path, dtype=type_np)
        elif file_ext == ".txt":
            data_np = np.loadtxt(input_file_path)
    except IOError:
        print("Error in loading files")
        return None

    return data_np




#############################Main function
if __name__ == "__main__":

    #Total Argument counts
    # argc = len(sys.argv) 
    # if argc < 4:
    #     sys.exit("Usage: fileName width height")

    data_0 = np.random.randint(0, 255,(10,10)).astype(np.uint8)
    data_0.tofile('test_data_in.bin')
    # #Load input argu
    file_name = sys.argv[1]
    # width = int(sys.argv[2])
    # height = int(sys.argv[3])

    # data_1 = np.array([500,50])

    # data_0 = np.array([[1, 0, 0], [0, 1, 0]])
    # data_1 = np.array([[1, 0, 1], [0, 1, 0]])
   
    # Compare_data(data_0, data_1)

    data = Load_files(file_name, 'uint8')
    print(data)
