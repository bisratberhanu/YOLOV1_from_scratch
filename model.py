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
        self.conv = nn.Conv2d(in_channels, out_channels, bias = False, **kwargs)
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.leakyRelu = nn.LeakyReLU(0.1)

    
    def forward(self, x):
        return self.leakyRelu(self.batchnorm(self.conv(x)))


class YOLOV1(nn.Module):
    def __init__(self, in_channels = 3, **kwargs):
        super(YOLOV1,self).__init__()
        self.architecture = architecture_config 
        self.in_channels = in_channels
        self.darknet = self.create_conv_layers(self.architecture)
        self.fcs = self._create_fcs(**kwargs)
    

    def forward(self,x):
        print("forward is being called here")
        x = self.darknet(x)
        return self.fcs(torch.flatten(x,start_dim =1))


    def create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels

        for x in architecture:
            if type(x) == tuple:
                layers += [CNNBlock(in_channels, x[1], kernel_size = x[0], stride = x[2],padding = x[3])]

                in_channels = x[1]
            elif type(x) == str:
                layers+= [nn.MaxPool2d(2 , 2)]
            

            elif type(x) == list:
                conv1, conv2, num_repeat = x[0], x[1], x[2]

                for i in range(num_repeat):
                    layers+= [CNNBlock(in_channels, conv1[1], kernel_size = conv1[0], stride =  conv1[2], padding = conv1[3])]
                    layers+= [CNNBlock(conv1[1], conv2[1], kernel_size = conv2[0], stride =  conv2[2], padding = conv2[3])]

                in_channels = conv2[1]
        print(type(nn.Sequential(*layers)))
        return nn.Sequential(*layers)

    def _create_fcs(self, split_size, num_boxes,num_classes):
        S,B,C = split_size, num_boxes, num_classes

        return nn.Sequential(nn.Flatten(), 
                        nn.Linear(S*S*1024, 496), 
                        nn.Dropout(0.0), 
                        nn.LeakyReLU(0.1),
                        nn.Linear(496, S*S*(C + B * 5)) )



def test(S= 7, B =2 , C = 20):
    model = YOLOV1(split_size= S, num_boxes = B, num_classes = C )
    X = torch.randn((2,3,448,448))
    print(X.shape)
    print(model(X).shape)
    




# test()

                    




