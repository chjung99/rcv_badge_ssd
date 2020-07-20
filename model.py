from torch import nn
from utils import *
import torch.nn.functional as F
from math import sqrt
import torchvision

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class VGGBase(nn.Module):
    def __init__(self):
        super(VGGBase,self).__init__()
#         torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros')
#         torch.nn.MaxPool2d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)
        
        self.conv1_1=nn.Conv2d(3,64,kernel_size=3,padding=1)
        self.conv1_2=nn.Conv2d(64,64,kernel_size=3,padding=1)
        
        self.maxpool_1=nn.MaxPool2d(kernel_size=2,stride=2)
        
        self.conv2_1=nn.Conv2d(64,128,kernel_size=3,padding=1)
        self.conv2_2=nn.Conv2d(128,128,kernel_size=3,padding=1)
        
        self.maxpool_2=nn.MaxPool2d(kernel_size=2,stride=2)
        
        self.conv3_1=nn.Conv2d(128,256,kernel_size=3,padding=1)
        self.conv3_2=nn.Conv2d(256,256,kernel_size=3,padding=1)
        self.conv3_3=nn.Conv2d(256,256,kernel_size=3,padding=1)
        
        self.maxpool_3=nn.MaxPool2d(kernel_size=2,stride=2,ceil_mode=True)
        
        self.conv4_1=nn.Conv2d(256,512,kernel_size=3,padding=1)
        self.conv4_2=nn.Conv2d(512,512,kernel_size=3,padding=1)
        self.conv4_3=nn.Conv2d(512,512,kernel_size=3,padding=1)
        
        self.maxpool_4=nn.MaxPool2d(kernel_size=2,stride=2)
        
        self.conv5_1=nn.Conv2d(512,512,kernel_size=3,padding=1)
        self.conv5_2=nn.Conv2d(512,512,kernel_size=3,padding=1)
        self.conv5_3=nn.Conv2d(512,512,kernel_size=3,padding=1)
        
        self.maxpool_5=nn.MaxPool2d(kernel_size=3,stride=1,padding=1)
        
        self.conv6=nn.Conv2d(512,1024,kernel_size=3,padding=6,dilation=6)
        
        self.conv7=nn.Conv2d(1024,1024,kernel_size=1)
        
        
        self.load_pretrained_layers()
    def forward(self,image):
        
        out=F.relu(self.conv1_1(image))
        out=F.relu(self.conv1_2(out))
        
        out=self.maxpool_1(out)
        
        out=F.relu(self.conv2_1(out))
        out=F.relu(self.conv2_2(out))
        
        out=self.maxpool_2(out)
        
        out=F.relu(self.conv3_1(out))
        out=F.relu(self.conv3_2(out))
        out=F.relu(self.conv3_3(out))
        
        out=self.maxpool_3(out)
        
        out=F.relu(self.conv4_1(out))
        out=F.relu(self.conv4_2(out))
        out=F.relu(self.conv4_3(out))
        
        conv4_3_feats=out
        
        out=self.maxpool_4(out)
        
        out=F.relu(self.conv5_1(out))
        out=F.relu(self.conv5_2(out))
        out=F.relu(self.conv5_3(out))
        
        out=self.maxpool_5(out)
        
        out=F.relu(self.conv6(out))
        out=F.relu(self.conv7(out))
        
        conv7_feats=out
        
        return conv4_3_feats,conv7_feats
    def load_pretrained_layers(self):
        state_dict=self.state_dict()

        names=list((state_dict).keys())
        

        
        pretrained_state_dict = torchvision.models.vgg16(pretrained=True).state_dict()
        pretrained_names = list(pretrained_state_dict.keys())

        
        for i, param in enumerate(names[:-4]): 
            state_dict[param] = pretrained_state_dict[pretrained_names[i]]
            
        self.load_state_dict(state_dict)
        
        print("base loaded!")
        
class AuxiliaryConvolutions(nn.Module):
    

    def __init__(self):
        super(AuxiliaryConvolutions, self).__init__()

        
        self.conv8_1 = nn.Conv2d(1024, 256, kernel_size=1, padding=0)  
        self.conv8_2 = nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)  
        self.conv9_1 = nn.Conv2d(512, 128, kernel_size=1, padding=0)
        self.conv9_2 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)  

        self.conv10_1 = nn.Conv2d(256, 128, kernel_size=1, padding=0)
        self.conv10_2 = nn.Conv2d(128, 256, kernel_size=3, padding=0)  

        self.conv11_1 = nn.Conv2d(256, 128, kernel_size=1, padding=0)
        self.conv11_2 = nn.Conv2d(128, 256, kernel_size=3, padding=0) 

       
        self.init_conv2d()

    def init_conv2d(self):
        
        for c in self.children():
            if isinstance(c, nn.Conv2d):
                nn.init.xavier_uniform_(c.weight)
                nn.init.constant_(c.bias, 0.)

    def forward(self, conv7_feats):
        
        out = F.relu(self.conv8_1(conv7_feats))  
        out = F.relu(self.conv8_2(out))  
        conv8_2_feats = out  
        out = F.relu(self.conv9_1(out))  
        out = F.relu(self.conv9_2(out))  
        conv9_2_feats = out  

        out = F.relu(self.conv10_1(out)) 
        out = F.relu(self.conv10_2(out))  
        conv10_2_feats = out 

        out = F.relu(self.conv11_1(out))  
        conv11_2_feats = F.relu(self.conv11_2(out))  

        
        return conv8_2_feats, conv9_2_feats, conv10_2_feats, conv11_2_feats

class PredictionConvolutions(nn.Module):
    def __init__(self):
        
        super(PredictionConvolutions, self).__init__()
        
#         n_boxes={'conv4_3'=4,
#                 'conv7'=6,
#                 'conv8_2'=6,
#                 'conv9_2'=6,
#                 'conv10_2'=4,
#                 'conv11_2'=4,
#                 }
        self.l_conv4_3=nn.Conv2d(512,4*4,kernel_size=3,padding=1)
        self.l_conv7=nn.Conv2d(1024,6*4,kernel_size=3,padding=1)
        self.l_conv8_2=nn.Conv2d(512,6*4,kernel_size=3,padding=1)
        self.l_conv9_2=nn.Conv2d(256,6*4,kernel_size=3,padding=1)
        self.l_conv10_2=nn.Conv2d(256,4*4,kernel_size=3,padding=1)
        self.l_conv11_2=nn.Conv2d(256,4*4,kernel_size=3,padding=1)
        
        self.c_conv4_3=nn.Conv2d(512,4*21,kernel_size=3,padding=1)
        self.c_conv7=nn.Conv2d(1024,6*21,kernel_size=3,padding=1)
        self.c_conv8_2=nn.Conv2d(512,6*21,kernel_size=3,padding=1)
        self.c_conv9_2=nn.Conv2d(256,6*21,kernel_size=3,padding=1)
        self.c_conv10_2=nn.Conv2d(256,4*21,kernel_size=3,padding=1)
        self.c_conv11_2=nn.Conv2d(256,4*21,kernel_size=3,padding=1)
        
        self.init_conv2d()
    def init_conv2d(self):
        
        for c in self.children():
            if isinstance(c, nn.Conv2d):
                nn.init.xavier_uniform_(c.weight)
                nn.init.constant_(c.bias, 0.)
#     def forward(self, conv4_3_feats, conv7_feats, conv8_2_feats, conv9_2_feats, conv10_2_feats, conv11_2_feats):
    def forward(self, conv4_3_feats):
        batch_size=conv4_3_feats.size(0)
        l_conv4_3=self.l_conv4_3(conv4_3_feats)#(N,16,38,38)
        l_conv4_3=l_conv4_3.permute(0,2,3,1)#(N,38,38,16)
        l_conv4_3=l_conv4_3.view(batch_size,-1,4)
        
        import pdb;pdb.set_trace()
        