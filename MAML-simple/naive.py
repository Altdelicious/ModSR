import torch
from torch import nn
from torch import optim
from torch import autograd
from torch.autograd import Variable
from torch.nn import functional as F

class Conv_ReLU_Block(nn.Module):
    def __init__(self):
        super(Conv_ReLU_Block, self).__init__()
        self.conv = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):
        return self.relu(self.conv(x))

class block(nn.Module):
    def __init__(self, features):
        super(block, self).__init__()
        self.features = features
        #self.bn1 = nn.BatchNorm2d(features)
        #self.bn2 = nn.BatchNorm2d(features)
        self.conv1 = nn.Conv2d(features, features, kernel_size = 1, bias=True)
        self.conv2 = nn.Conv2d(features, features, kernel_size = 1, bias=True)
        self.r = nn.ReLU(inplace=True)
        
    def forward(self, z):
        op = self.r(self.conv1(z))
        op = self.r(self.conv2(op))
        
        #print(op.size())

        # multiply spatial weights
        out = op + z
        return out


class VDSR(nn.Module):
    def __init__(self):
        super(VDSR, self).__init__()
        
        self.res_1, self.res_2, self.res_3 = Conv_ReLU_Block(), Conv_ReLU_Block(), Conv_ReLU_Block()
        self.res_4, self.res_5, self.res_6 = Conv_ReLU_Block(), Conv_ReLU_Block(), Conv_ReLU_Block()
        self.res_7, self.res_8, self.res_9 = Conv_ReLU_Block(), Conv_ReLU_Block(), Conv_ReLU_Block()
        self.res_10, self.res_11, self.res_12 = Conv_ReLU_Block(), Conv_ReLU_Block(), Conv_ReLU_Block()
        self.res_13, self.res_14, self.res_15 = Conv_ReLU_Block(), Conv_ReLU_Block(), Conv_ReLU_Block()
        self.res_16, self.res_17, self.res_18 = Conv_ReLU_Block(), Conv_ReLU_Block(), Conv_ReLU_Block()
        
        self.input = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.output = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)
        
        self.criteon = nn.L1Loss()
        
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, target):
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

        loss = self.criteon(out, target)

        return loss, out


class VDSR_mod(nn.Module):
    def __init__(self):
        super(VDSR_mod, self).__init__()
        self.input = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.output = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)
        
        self.res_1, self.res_2, self.res_3 = Conv_ReLU_Block(), Conv_ReLU_Block(), Conv_ReLU_Block()
        self.res_4, self.res_5, self.res_6 = Conv_ReLU_Block(), Conv_ReLU_Block(), Conv_ReLU_Block()
        self.res_7, self.res_8, self.res_9 = Conv_ReLU_Block(), Conv_ReLU_Block(), Conv_ReLU_Block()
        self.res_10, self.res_11, self.res_12 = Conv_ReLU_Block(), Conv_ReLU_Block(), Conv_ReLU_Block()
        self.res_13, self.res_14, self.res_15 = Conv_ReLU_Block(), Conv_ReLU_Block(), Conv_ReLU_Block()
        self.res_16, self.res_17, self.res_18 = Conv_ReLU_Block(), Conv_ReLU_Block(), Conv_ReLU_Block()
        
        self.block_1, self.block_2, self.block_3 = block(64), block(64), block(64)
        self.block_4, self.block_5, self.block_6 = block(64), block(64), block(64)
        self.block_7, self.block_8, self.block_9 = block(64), block(64), block(64)
        self.block_10, self.block_11, self.block_12 = block(64), block(64), block(64)
        self.block_13, self.block_14, self.block_15 = block(64), block(64), block(64)
        self.block_16, self.block_17, self.block_18 = block(64), block(64), block(64)
        
        self.relu = nn.ReLU(inplace=True)
        
        self.criteon = nn.L1Loss()
        #self.dummy_param = nn.Parameter(torch.empty(0))

    def forward(self, x, target):
        residual = x
        out = self.relu(self.input(x))
        
        out = self.block_1(self.res_1(out))
        out = self.block_2(self.res_2(out))
        out = self.block_3(self.res_3(out))
        out = self.block_4(self.res_4(out))
        out = self.block_5(self.res_5(out))
        out = self.block_6(self.res_6(out))
        out = self.block_7(self.res_7(out))
        out = self.block_8(self.res_8(out))
        out = self.block_9(self.res_9(out))
        out = self.block_10(self.res_10(out))
        out = self.block_11(self.res_11(out))
        out = self.block_12(self.res_12(out))
        out = self.block_13(self.res_13(out))
        out = self.block_14(self.res_14(out))
        out = self.block_15(self.res_15(out))
        out = self.block_16(self.res_16(out))
        out = self.block_17(self.res_17(out))
        out = self.block_18(self.res_18(out))
        
        out = self.output(out)
        out = torch.add(out,residual)
        
        loss = self.criteon(out, target)

        return loss, out
