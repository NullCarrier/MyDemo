import os
import urllib
import traceback
import time
import sys
import numpy as np
#import cv2 as cv
import torch
import torch.onnx

import onnxruntime as rt
#from rknn.api import RKNN
from numpy.linalg import norm


def Cal_similarity(a, b):
    return ( np.inner(a, b) / ( norm(a) * norm(b) ) )



def Check_results(inputData_A, inputData_B):
    print("----------------------------------------")
    #Load input data as float type
    rslt_A = np.fromfile(inputData_A, dtype=np.float32)
    #rslt_A = rslt_A.reshape(80,80,3).transpose((2,0,1)).flatten() 
    print("The first input data type: ", rslt_A.dtype)

    rslt_B = np.fromfile(inputData_B, dtype=np.float32)
    # rslt_B = np.load(inputData_B)
    # rslt_B = rslt_B.flatten()
    print("The second input data type: ", rslt_B.dtype)

    print("The shape of first input: ", rslt_A.shape)
    print("The shape of second input: ", rslt_B.shape)
    if rslt_A.shape != rslt_B.shape:
        print("Error: two inputs have different shape")
        return -1;

    cos_sim  = Cal_similarity(rslt_A, rslt_B) 
    #cos_sim  = check_outputs(rslt_sim, rslt_rknn_fp32)
    print("The similarity: ", cos_sim)
    print("----------------------------------------")



def ignore_dim_with_zero(_shape, _shape_target):
    _shape = list(_shape)
    _shape_target = list(_shape_target)
    for i in range(_shape.count(1)):
        _shape.remove(1)
    for j in range(_shape_target.count(1)):
        _shape_target.remove(1)
    if _shape == _shape_target:
        return True
    else:
        return False

def Get_random_data_nchw(): 
    data_list = [None] * 6        #n c    h   w
    data_list[0] = np.random.randint(256, size=(1, 96, 48, 80)).astype(np.int8)
    data_list[1] = np.random.randint(256, size=(1, 64, 24, 40)).astype(np.int8)
    data_list[2] = np.random.randint(256, size=(1, 160, 12, 20)).astype(np.int8)
    data_list[3] = np.random.randint(256, size=(1, 96, 6, 10)).astype(np.int8)
    data_list[4] = np.random.randint(256, size=(1, 32, 48, 80)).astype(np.int8)
    data_list[5] = np.random.randint(256, size=(1, 32, 96, 160)).astype(np.int8)

    return data_list

def Get_rand_data_nc1hwc1():
    data_list = [None] * 6
    data_list[0] = np.random.randint(256,size=(1, 6, 48, 80, 16)).astype(np.int8)
    print("The first input shape: ", data_list[0].shape)
    data_list[1] = np.random.randint(256, size=(1, 4, 24, 40, 16)).astype(np.int8)
    data_list[2] = np.random.randint(256,size=(1, 10, 12, 20, 16)).astype(np.int8)
    data_list[3] = np.random.randint(256,size=(1, 6, 6, 10, 16)).astype(np.int8)
    data_list[4] = np.random.randint(256, size=(1, 2, 48, 80, 16)).astype(np.int8)
    data_list[5] = np.random.randint(256, size=(1, 2, 96, 160, 16)).astype(np.int8)

    return data_list
    


def Export_2Onnx(torch_model, model_save_path):


    # set the model to inference mode
    torch_model.eval()
    # Input to the model
    input_rand = torch.randn(1, 3, 224, 224 ) # nchw
    torch_out = torch_model(input_rand)

    # Export the model
    torch.onnx.export(torch_model,               # model being run
                  input_rand,                         # model input (or a tuple for multiple inputs)
                  model_save_path,   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=12,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output']) # the model's output names
                #   dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                #                 'output' : {0 : 'batch_size'}})


    
        

def Get_Models_From_Torchvision(model_list = ''):
    import torchvision.models as models

    if not model_list:
        return 

    # dict_models = {}
    # for i in range(len(model_list)):
    #     dict_models
    dump_model_path = './mobilenet_series/'

    for model in model_list:

        if model == 'mobilenet_v2':
            mobilenet_v2 = models.mobilenet_v2(pretrained=True)
            full_model_path = 'mobilenet_v2' + '.onnx' 
            Export_2Onnx(mobilenet_v2, full_model_path)
            # dict_models.update({'mobilenet_v2' : mobilenet_v2})
        elif model == 'mobilenet_v1':
            mobilenet_v1 = models.mobilenet_v1(pretrained=True)
            # dict_models.update({'mobilenet_v1' : mobilenet_v1})
        elif model == 'resnet50': 
            resnet50 = models.resnet50(pretrained=True)
            full_model_path = dump_model_path + 'resnet50' + '.onnx' 
            Export_2Onnx(resnet50, full_model_path)
            # dict_models.update({'resnet50' : resnet50})
        elif model == 'mobilenet_v3':
            mobilenet_v3 = models.mobilenet_v3_small(pretrained=True)
            full_model_path = dump_model_path + 'mobilenet_v3_small' + '.onnx' 
            Export_2Onnx(mobilenet_v3, full_model_path)
            # dict_models.update({'mobilenet_v3' : mobilenet_v3})

    



    

ANALISYS_ON=False
DATASET='./dataset.txt'
CROP=False
USE_ONNX=True


if __name__ == '__main__':

    if len(sys.argv) != 3:
        print("Usage: python3 onnx2rknn.py xxx xxx.rknn")
        exit(1)
    PT_MODEL = ''
    ONNX_MODEL = ''
    if USE_ONNX: 
        ONNX_MODEL = sys.argv[1]
    else:
        PT_MODEL = sys.argv[1]

    RKNN_MODEL = sys.argv[2]

    #Create RKNN object
    rknn = RKNN(verbose='Debug')

    # pre-process config
    print('--> Config model')
    
    # rknn.config(target_platform='rk3588', mean_values=mean_input, std_values=std_input)
    rknn.config(target_platform='rk3562')
    print('done')

    # Load ONNX model
    print('--> Loading model')
    if USE_ONNX:
        if CROP:
            ret = rknn.load_onnx(model=ONNX_MODEL, inputs=['1695', '1737'],  
                            input_size_list=[[1,3,10,10,51], [1,3,10,10,17]], outputs=['1812'])
        else:
            ret = rknn.load_onnx(model=ONNX_MODEL)
    else:
        ret = rknn.load_pytorch(model=PT_MODEL, input_size_list=[[1,3,480,640]])
    if ret != 0:
        print('Load model failed!')
        exit(ret)
    print('done')

    # num_input = 2
    # data_for_quantization = [None] * num_input
    # for data_entry in data_for_quantization:
    #     data_entry = np.random.randint(256,size=(1,3,10,10,51)).astype(np.uint8)
    # for i in range(num_input):
    #     out_fileName = 'data_nchw_' + str(i)  
    #     np.save(out_fileName, data_for_quantization[i]) 
 
    # img = cv.imread('./img_256x256_test.png')
    # img_rs = cv.resize(img, (640, 480))
    # # img = cv.imread('./dog_224x224.jpg')

    # img_rgb = cv.cvtColor(img_rs, cv.COLOR_BGR2RGB)
    # img_rgb_nchw = img_rgb.transpose((2,0,1))

    # out_fileName = 'data_nchw_' + str(0)  
    # # np.save(out_fileName, img_rgb_nchw) 

    # data_in__qnt_0 = np.random.randint(256,size=(1,3,10,10,51)).astype(np.int8)
    # np.save(out_fileName, data_in__qnt_0) 
    # data_in_qnt_1 = np.random.randint(256,size=(1,3,10,10,17)).astype(np.int8)
    out_fileName = 'data_nchw_' + str(1)  
    # np.save(out_fileName, data_in_qnt_1) 
    data_ram = np.random.rand(1,16,3600).astype(np.float32)
    np.save(out_fileName, data_ram) 

    # Build model
    print('--> Building model')
    ret = rknn.build(do_quantization=False, dataset=DATASET)
    if ret != 0:
        print('Build model failed!')
        exit(ret)
    print('done')


    #==========================================================================================
    if ANALISYS_ON:
    # Accuracy analysis
        print('--> Accuracy analysis')
        data_in = np.random.randint(256,size=()).astype(np.int8)
        data_in_0 = np.random.rand(1,3,10,10,51).astype(np.float32)
        # data_in_1 = np.random.rand(1,3,10,10,17).astype(np.float32)

        # img = cv.imread('./bus.jpg')
        # img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        # img_rgb_nchw = img_rgb.transpose((2,0,1))

        Ret = rknn.accuracy_analysis(inputs=[data_in, data_in_0], 
                                    target='rk3562', output_dir='./analysis_outputs')
        if ret != 0:
            print('Accuracy analysis failed!')
            exit(ret)
        print('done')

    #Export RKNN model
    print('--> Export rknn model')
    ret = rknn.export_rknn(RKNN_MODEL)
    if ret != 0:
        print('Export rknn model failed!')
        exit(ret)
    print('done')
    #==========================================================================================

    rknn.release()
