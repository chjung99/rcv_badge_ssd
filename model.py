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
                
    def forward(self, conv4_3_feats, conv7_feats, conv8_2_feats, conv9_2_feats, conv10_2_feats, conv11_2_feats):

        batch_size=conv4_3_feats.size(0)
        
        l_conv4_3=self.l_conv4_3(conv4_3_feats)#(N,16,38,38)
        l_conv4_3=l_conv4_3.permute(0,2,3,1).contiguous()#(N,38,38,16)
        l_conv4_3=l_conv4_3.view(batch_size,-1,4)
        #contiguous=>for #RuntimeError: view size is not compatible with input tensor's size and stride (at least one dimension spans across two contiguous subspaces). Use .reshape(...) instead.
        l_conv7=self.l_conv7(conv7_feats)#(N,24,10,10)
        l_conv7=l_conv7.permute(0,2,3,1).contiguous()#(N,10,10,24)
        l_conv7=l_conv7.view(batch_size,-1,4)
        
        l_conv8_2=self.l_conv8_2(conv8_2_feats)#(N,24,10,10)
        l_conv8_2=l_conv8_2.permute(0,2,3,1).contiguous()#(N,10,10,24)
        l_conv8_2=l_conv8_2.view(batch_size,-1,4)
        
        l_conv9_2=self.l_conv9_2(conv9_2_feats)#(N,24,10,10)
        l_conv9_2=l_conv9_2.permute(0,2,3,1).contiguous()#(N,10,10,24)
        l_conv9_2=l_conv9_2.view(batch_size,-1,4)
        
        l_conv10_2=self.l_conv10_2(conv10_2_feats)#(N,24,10,10)
        l_conv10_2=l_conv10_2.permute(0,2,3,1).contiguous()#(N,10,10,24)
        l_conv10_2=l_conv10_2.view(batch_size,-1,4)
        
        l_conv11_2=self.l_conv11_2(conv11_2_feats)#(N,24,10,10)
        l_conv11_2=l_conv11_2.permute(0,2,3,1).contiguous()#(N,10,10,24)
        l_conv11_2=l_conv11_2.view(batch_size,-1,4)
        
        
        c_conv4_3=self.c_conv4_3(conv4_3_feats)#(N,4*21,38,38)
        c_conv4_3=c_conv4_3.permute(0,2,3,1).contiguous()#(N,38,38,16)
        c_conv4_3=c_conv4_3.view(batch_size,-1,21)
        
        c_conv7=self.c_conv7(conv7_feats)#(N,24,10,10)
        c_conv7=c_conv7.permute(0,2,3,1).contiguous()#(N,10,10,24)
        c_conv7=c_conv7.view(batch_size,-1,21)
        
        c_conv8_2=self.c_conv8_2(conv8_2_feats)#(N,24,10,10)
        c_conv8_2=c_conv8_2.permute(0,2,3,1).contiguous()#(N,10,10,24)
        c_conv8_2=c_conv8_2.view(batch_size,-1,21)
        
        c_conv9_2=self.c_conv9_2(conv9_2_feats)#(N,24,10,10)
        c_conv9_2=c_conv9_2.permute(0,2,3,1).contiguous()#(N,10,10,24)
        c_conv9_2=c_conv9_2.view(batch_size,-1,21)
        
        c_conv10_2=self.c_conv10_2(conv10_2_feats)#(N,24,10,10)
        c_conv10_2=c_conv10_2.permute(0,2,3,1).contiguous()#(N,10,10,24)
        c_conv10_2=c_conv10_2.view(batch_size,-1,21)
        
        c_conv11_2=self.c_conv11_2(conv11_2_feats)#(N,24,10,10)
        c_conv11_2=c_conv11_2.permute(0,2,3,1).contiguous()#(N,10,10,24)
        c_conv11_2=c_conv11_2.view(batch_size,-1,21)
        
        locs=torch.cat([l_conv4_3, l_conv7, l_conv8_2, l_conv9_2, l_conv10_2, l_conv11_2],dim=1)
        classes_scores=torch.cat([c_conv4_3, c_conv7, c_conv8_2, c_conv9_2, c_conv10_2, c_conv11_2],dim=1)
        
        return locs,classes_scores
        
class SSD300(nn.Module):
    def __init__(self):
        super(SSD300,self).__init__()
        self.base=VGGBase()
        self.aux=AuxiliaryConvolutions()
        self.pred=PredictionConvolutions()
        self.priors_cxcy=self.create_prior_boxes()
    def forward(self,image):
        conv4_3,conv7=self.base(image)
        conv8_2,conv9_2,conv10_2,conv11_2=self.aux(conv7)
        locs,classes_scores=self.pred(conv4_3,conv7,conv8_2,conv9_2,conv10_2,conv11_2)#(N,8732,4),(N,8732,21)
        
        return locs,classes_scores
    def create_prior_boxes(self):
        fmaps_dims = {'conv4_3': 38,
                     'conv7': 19,
                     'conv8_2': 10,
                     'conv9_2': 5,
                     'conv10_2': 3,
                     'conv11_2': 1}

        obj_scales = {'conv4_3': 0.1,
                      'conv7': 0.2,
                      'conv8_2': 0.375,
                      'conv9_2': 0.55,
                      'conv10_2': 0.725,
                      'conv11_2': 0.9}

        aspect_ratios = {'conv4_3': [1., 2., 0.5],
                         'conv7': [1., 2., 3., 0.5, .333],
                         'conv8_2': [1., 2., 3., 0.5, .333],
                         'conv9_2': [1., 2., 3., 0.5, .333],
                         'conv10_2': [1., 2., 0.5],
                         'conv11_2': [1., 2., 0.5]}

        fmaps = list(fmaps_dims.keys())

        prior_boxes = []
        
        for k,fmap in enumerate(fmaps):
            for i in range(fmaps_dims[fmap]):
                for j in range(fmaps_dims[fmap]):
                    cx=(j+0.5)/fmaps_dims[fmap]
                    cy=(i+0.5)/fmaps_dims[fmap]
                    for ratio in aspect_ratios[fmap]:
                        w=obj_scales[fmap]*sqrt(ratio)
                        h=obj_scales[fmap]/sqrt(ratio)
                        prior_boxes.append([cx,cy,w,h])
                        if ratio == 1:
                            try:
                                w=sqrt(obj_scales[fmaps][k]*obj_scales[fmaps][k+1])
                                h=w
                                
                            except:
                                
                                w=1
                                h=1
                            prior_boxes.append([cx,cy,w,h])
        prior_boxes=torch.FloatTensor(prior_boxes).to(device)
        prior_boxes=prior_boxes.clamp_(0,1)
        
        return prior_boxes
    
#     def detect_objects(self,predicted_locs,predicted_scores,min_score,max_overlap,top_k):

class MultiBoxLoss(nn.Module):
    def __init__(self,priors_cxcy,threshold=0.5,neg_pos_ratio=3,alpha=1):
        super(MultiBoxLoss,self).__init__()
        self.priors_cxcy=priors_cxcy
        self.priors_xy=cxcy_to_xy(self.priors_cxcy)
        
        self.l1Loss=nn.L1Loss()
        self.cross_entropy=nn.CrossEntropyLoss()
        
    def forward(self,pred_locs,pred_scores,boxes,labels):
        
        
            
        
        
        
        
        conf_loss_all=
        conf_loss_pos=
        conf_loss_neg=
        
        
        return conf_loss + alpha*loc_loss
    
        
        