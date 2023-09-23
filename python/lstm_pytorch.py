'''
Author: Chao Li
Date: 2022-07-22 17:42:22
LastEditTime: 2022-07-28 11:08:33
Editors: Chao
Description: test lstm
'''

from tkinter import N
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.onnx
import numpy as np

kernel_size = 1
class mynet(nn.Module):
    def __init__(self):
        super(mynet, self).__init__()
        self.lstm = nn.LSTM(input_size=256,hidden_size=64)
        self.conv1 = nn.Conv2d(3,2,(3,3),padding=[1,1])
    def forward(self, x0, h0,c0):
        x0 = self.conv1(x0)
        x = torch.reshape(x0,(64,1,256))
        y =  self.lstm(x,(h0,c0))
        return y
net = mynet()
net.eval()
x=torch.rand(1,3,32,256)
h0 = torch.rand(1,1,64)
c0 = torch.rand(1,1,64)
torch.onnx.export(net, (x,h0,c0), "lstm.onnx", verbose=True, input_names=['input1'], output_names=['output'])


def weight_init(net):
    # 递归获得net的所有子代Module
    for op in net.modules():
        # 针对不同类型操作采用不同初始化方式
        if isinstance(op, nn.ConvTranspose2d) :
            weight1 = np.load("convTranspose_weight.npy")
            op.weight.data = torch.from_numpy(weight1)
        # 这里可以对Conv等操作进行其它方式的初始化
        elif isinstance(op,nn.InstanceNorm2d):
            print("aaa")
            weight = np.load("model_2_weight.npy")
            bias = np.load("model_2_bias.npy")
            op.weight.data = torch.from_numpy(weight)
            op.bias.data = torch.from_numpy(bias)
        else:
            pass


if __name__ == "__main__":
    with torch.no_grad():
        net = mynet()
        # weight_init(net)
        torch.save(net.state_dict(), "lstm.pth")
        net.load_state_dict(torch.load("lstm.pth", map_location="cpu"))
        net.eval()
        output = net(x,h0,c0)
        output = output.numpy()
        print("pytorch res")
        print(output)
        np.savetxt("torch_res.txt",output.flatten(),fmt="%.8f",delimiter="\n")

        # trace_model = torch.jit.trace(model, torch.Tensor(1,1,2,2))
        # trace_model.save('./resize_bilinear_align_corners.pt')
        torch.onnx.export(net,(x,h0,c0),"lstm.onnx")


