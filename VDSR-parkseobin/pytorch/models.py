import torch
import torch.nn as nn
from math import sqrt


class Conv_ReLU_Block(nn.Module):
    def __init__(self):
        super(Conv_ReLU_Block, self).__init__()
        self.conv = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        

    def forward(self, x):
        return self.relu(self.conv(x))



class VDSR(nn.Module):
    def __init__(self, rgb=False):
        super(VDSR, self).__init__()
        
        self.res_1, self.res_2, self.res_3 = Conv_ReLU_Block(), Conv_ReLU_Block(), Conv_ReLU_Block()
        self.res_4, self.res_5, self.res_6 = Conv_ReLU_Block(), Conv_ReLU_Block(), Conv_ReLU_Block()
        self.res_7, self.res_8, self.res_9 = Conv_ReLU_Block(), Conv_ReLU_Block(), Conv_ReLU_Block()
        self.res_10, self.res_11, self.res_12 = Conv_ReLU_Block(), Conv_ReLU_Block(), Conv_ReLU_Block()
        self.res_13, self.res_14, self.res_15 = Conv_ReLU_Block(), Conv_ReLU_Block(), Conv_ReLU_Block()
        self.res_16, self.res_17, self.res_18 = Conv_ReLU_Block(), Conv_ReLU_Block(), Conv_ReLU_Block()
        
        self.input = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.output = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)
        
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.relu(self.input(x))
        
        out = self.res_1(out)
        out = self.res_2(out)
        out = self.res_3(out)
        out = self.res_4(out)
        out = self.res_5(out)
        out = self.res_6(out)
        out = self.res_7(out)
        out = self.res_8(out)
        out = self.res_9(out)
        out = self.res_10(out)
        out = self.res_11(out)
        out = self.res_12(out)
        out = self.res_13(out)
        out = self.res_14(out)
        out = self.res_15(out)
        out = self.res_16(out)
        out = self.res_17(out)
        out = self.res_18(out)
        
        out = self.output(out)
        out = torch.add(out,residual)

        return out
