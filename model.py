import torch
import torch.nn as nn


#list : last value is how many time  it be repeated
 #kernel,feature, stride,padding [
architecture_config = [                                                
        (7,64,2,3),
        "M", 
        (3,192,1,1),
        "M",
        (1,128,1,0),
        (3,256,1,1),
        (1,256,1,0),
        (3,512,1,1),
        "M",
        [(1,256,1,0), (3,512,1,1),4],
        (1,512,1,0),
        (3,1024,1,1),
        "M",
        [(1,512,1,0),(3,1024,1,1),2],
        (3,1024,2,1),
        (3,1024,1,1),
        (3,1024,1,1),
        (3,1024,1,1)]  #kernel,feature, stride,padding [

class CNNBlock(nn.Module):
    def __init__(self,in_channels, out_channels, **kwargs):
        super(CNNBlock, self).__init__()
        self.conv = nn.conv2d(in_channels, out_channels, bias = False, **kwargs)

