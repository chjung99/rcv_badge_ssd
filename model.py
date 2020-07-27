from torch import nn
from utils import *
import torch.nn.functional as F
from math import sqrt
import torchvision

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class VGGBase(nn.Module):
    def __init__(self):
        super(VGGBase,self).__init__()
        
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
        
        out=(F.relu(self.conv6(out)))
        out=(F.relu(self.conv7(out)))
        
        conv7_feats=out
        
        return conv4_3_feats,conv7_feats
    def load_pretrained_layers(self):
        state_dict=self.state_dict()

        names=list((state_dict).keys())
        

        
        pretrained_state_dict = torchvision.models.vgg16(pretrained=True).state_dict()
        pretrained_names = list(pretrained_state_dict.keys())

        
        for i, param in enumerate(names[:-4]): 
            state_dict[param] = pretrained_state_dict[pretrained_names[i]]
        
        conv_fc6_weight = pretrained_state_dict['classifier.0.weight'].view(4096, 512, 7, 7)  # (4096, 512, 7, 7)
        conv_fc6_bias = pretrained_state_dict['classifier.0.bias']  # (4096)
        state_dict['conv6.weight'] = decimate(conv_fc6_weight, m=[4, None, 3, 3])  # (1024, 512, 3, 3)
        state_dict['conv6.bias'] = decimate(conv_fc6_bias, m=[4])  # (1024)
        # fc7
        conv_fc7_weight = pretrained_state_dict['classifier.3.weight'].view(4096, 4096, 1, 1)  # (4096, 4096, 1, 1)
        conv_fc7_bias = pretrained_state_dict['classifier.3.bias']  # (4096)
        state_dict['conv7.weight'] = decimate(conv_fc7_weight, m=[4, 4, None, None])  # (1024, 1024, 1, 1)
        state_dict['conv7.bias'] = decimate(conv_fc7_bias, m=[4])  # (1024)

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
        
#         self.bn8_1=nn.BatchNorm2d(256)
#         self.bn8_2=nn.BatchNorm2d(512)
        
#         self.bn9_1=nn.BatchNorm2d(128)
#         self.bn9_2=nn.BatchNorm2d(256)
        
#         self.bn10_1=nn.BatchNorm2d(128)
#         self.bn10_2=nn.BatchNorm2d(256)
        
#         self.bn11_1=nn.BatchNorm2d(128)
#         self.bn11_2=nn.BatchNorm2d(256)
       
        self.init_conv2d()

    def init_conv2d(self):
        
        for c in self.children():
            if isinstance(c, nn.Conv2d):
                nn.init.xavier_uniform_(c.weight)
                nn.init.constant_(c.bias, 0.)

    def forward(self, conv7_feats):
        
#         out = self.bn8_1(F.relu(self.conv8_1(conv7_feats))) 
#         out = self.bn8_2(F.relu(self.conv8_2(out)))
#         conv8_2_feats = out  
#         out = self.bn9_1(F.relu(self.conv9_1(out)))
#         out = self.bn9_2(F.relu(self.conv9_2(out)))
#         conv9_2_feats = out  

#         out = self.bn10_1(F.relu(self.conv10_1(out)))
#         out = self.bn10_2(F.relu(self.conv10_2(out)))
#         conv10_2_feats = out 

#         out = self.bn11_1(F.relu(self.conv11_1(out)))
#         conv11_2_feats = self.bn11_2(F.relu(self.conv11_2(out)))
        out = (F.relu(self.conv8_1(conv7_feats))) 
        out = (F.relu(self.conv8_2(out)))
        conv8_2_feats = out  
        out = (F.relu(self.conv9_1(out)))
        out = (F.relu(self.conv9_2(out)))
        conv9_2_feats = out  

        out = (F.relu(self.conv10_1(out)))
        out = (F.relu(self.conv10_2(out)))
        conv10_2_feats = out 

        out = (F.relu(self.conv11_1(out)))
        conv11_2_feats = (F.relu(self.conv11_2(out)))
        
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
        
        self.rescale_factors = nn.Parameter(torch.FloatTensor(1, 512, 1, 1))  # there are 512 channels in conv4_3_feats
        nn.init.constant_(self.rescale_factors, 20)
        
        self.priors_cxcy=self.create_prior_boxes()
    def forward(self,image):
        conv4_3,conv7=self.base(image)
        conv8_2,conv9_2,conv10_2,conv11_2=self.aux(conv7)
        
        norm = conv4_3.pow(2).sum(dim=1, keepdim=True).sqrt()  # (N, 1, 38, 38)
        conv4_3 = conv4_3 / norm  # (N, 512, 38, 38)
        conv4_3 = conv4_3 * self.rescale_factors  # (N, 512, 38, 38)
        
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
                        if ratio == 1.:
                            try:
                                
                                w=sqrt(obj_scales[fmaps[k]]*obj_scales[fmaps[k+1]])
                                h=w
                                
                            except:
                                
#                                 w=sqrt(obj_scales[fmaps[k]]*0.875)
#                                 h=w
                                w=1
                                h=1
                            prior_boxes.append([cx,cy,w,h])

        prior_boxes=torch.FloatTensor(prior_boxes).to(device)
        prior_boxes=prior_boxes.clamp_(0,1)
        
        return prior_boxes

    def detect_objects(self,predicted_locs,predicted_scores,min_score,max_overlap,top_k):
        batch_size = predicted_locs.size(0)
        
        predicted_scores = F.softmax(predicted_scores, dim=2)  # (N, 8732, 21)

        
        all_images_boxes = list()
        all_images_labels = list()
        all_images_scores = list()

        

        for i in range(batch_size):
            
            decoded_locs = cxcy_to_xy(
                gcxgcy_to_cxcy(predicted_locs[i], self.priors_cxcy))  # (8732, 4)

            
            image_boxes = list()
            image_labels = list()
            image_scores = list()

            max_scores, best_label = predicted_scores[i].max(dim=1)  # (8732)

            
            for c in range(1, 21):
                
                class_scores = predicted_scores[i][:, c]  # (8732)
                score_above_min_score = class_scores > min_score  
                n_above_min_score = score_above_min_score.sum().item()
                if n_above_min_score == 0:
                    continue
                class_scores = class_scores[score_above_min_score]  
                class_decoded_locs = decoded_locs[score_above_min_score] 

                
                class_scores, sort_ind = class_scores.sort(dim=0, descending=True) 
                class_decoded_locs = class_decoded_locs[sort_ind]  

                
                overlap = find_jaccard_overlap(class_decoded_locs, class_decoded_locs)  
                # (NMS)

                
                suppress = torch.zeros((n_above_min_score), dtype=torch.uint8).to(device)  

               
                for box in range(class_decoded_locs.size(0)):
                    
                    
                    if suppress[box] == 1:
                        continue

                    
                    suppress = torch.max(suppress, (overlap[box] > max_overlap).byte())
                    
                    suppress[box] = 0
                suppress=suppress.bool()

                image_boxes.append(class_decoded_locs[~suppress])
                
                image_labels.append(torch.LongTensor((~suppress).sum().item() * [c]).to(device))
                image_scores.append(class_scores[~suppress])

           
            if len(image_boxes) == 0:
                image_boxes.append(torch.FloatTensor([[0., 0., 1., 1.]]).to(device))
                image_labels.append(torch.LongTensor([0]).to(device))
                image_scores.append(torch.FloatTensor([0.]).to(device))

         
            image_boxes = torch.cat(image_boxes, dim=0)  
            image_labels = torch.cat(image_labels, dim=0)  
            image_scores = torch.cat(image_scores, dim=0)  
            n_objects = image_scores.size(0)

            
            if n_objects > top_k:
                image_scores, sort_ind = image_scores.sort(dim=0, descending=True)
                image_scores = image_scores[:top_k]  
                image_boxes = image_boxes[sort_ind][:top_k]  
                image_labels = image_labels[sort_ind][:top_k]  

            
            all_images_boxes.append(image_boxes)
            all_images_labels.append(image_labels)
            all_images_scores.append(image_scores)

        return all_images_boxes, all_images_labels, all_images_scores  


class MultiBoxLoss(nn.Module):
    def __init__(self,priors_cxcy,threshold=0.5,neg_pos_ratio=3,alpha=1):
        super(MultiBoxLoss,self).__init__()
        self.priors_cxcy=priors_cxcy
        self.priors_xy=cxcy_to_xy(self.priors_cxcy)
        self.threshold=threshold
        self.l1Loss=nn.SmoothL1Loss(reduce=False,reduction='none')
        self.cross_entropy=nn.CrossEntropyLoss(reduce=False,reduction='none')
        self.alpha=alpha
        
    def forward(self,pred_locs,pred_scores,boxes,labels):
        
        batch_size=pred_locs.size(0)
        gt_locs = torch.zeros((batch_size, 8732, 4), dtype=torch.float).to(device)  # (N, 8732, 4)
        gt_labels = torch.zeros((batch_size, 8732), dtype=torch.long).to(device)  # (N, 8732)
        #여기서부터 하는 과정들은 모두 8732개 박스를 고려해주기 위한 shape 맞춤정도로 이해
        for i in range(batch_size):
            n_objects=boxes[i].size(0)
            overlap=find_jaccard_overlap(boxes[i],self.priors_xy)#(n_obj,8732)
            overlap_each_prior,object_each_prior=overlap.max(dim=0)#(8732)object_each_prior에는 최대 n_object 만큼 무작위 배치될수있고 최소 1개 배치될 수 있다
            _,prior_each_object=overlap.max(dim=1)#(n_obj)
            
            object_each_prior[prior_each_object] = torch.LongTensor(range(n_objects)).to(device)#(n_obj)여기서 object_each_prior에 최소  n_object 만큼 무작위 배치한다.그러면 8732중에 적어도 n_object를 대표하는 박스가 생기게된다
            overlap_each_prior[prior_each_object] = 1.
            label_each_prior=labels[i][object_each_prior]#(8732)label
            label_each_prior[overlap_each_prior<self.threshold]=0#background
            
            gt_labels[i]=label_each_prior#8732개 박스에 임의의 labeling but 반드시 정답 포함
            gt_locs[i]=cxcy_to_gcxgcy(xy_to_cxcy(boxes[i][object_each_prior]),self.priors_cxcy)#모델은 prior를 얼마나 움직일지를 예측하므로 
        
        pos=(gt_labels!=0)#positive

        if(pos.sum()!=0):
            
            n_pos=pos.sum(dim=1)
            n_hard_neg=n_pos*3
            conf_loss_all=self.cross_entropy(pred_scores.view(-1,21),gt_labels.view(-1))#(N,8732,21)(N,8732)->(N*8732,21)(N*8732)#reduce=False=>(1)x (N*8732)o각각의 prior의 loss 확인 가능
            conf_loss_all = conf_loss_all.view(batch_size, 8732)
            conf_pos_all=0
            conf_neg_all=0
            loc_all=0
            total_loss=0
            for i in range(batch_size): 
                loc_loss=self.l1Loss(pred_locs[i][pos[i]],gt_locs[i][pos[i]])#(8732,4)
                conf_loss_pos=conf_loss_all[i][pos[i]]
                conf_loss_neg=conf_loss_all[i].clone()
                conf_loss_neg[pos[i]]=0.
                conf_loss_neg,_=conf_loss_neg.sort(descending=True)
                hard_neg=torch.LongTensor(range(8732)).to(device)
                hard_neg=hard_neg<n_hard_neg[i].unsqueeze(0)
                conf_loss_neg=conf_loss_neg[hard_neg]
                conf_loss=(conf_loss_pos.sum()+conf_loss_neg.sum())
            
                conf_pos_all+=conf_loss_pos.sum()
                conf_neg_all+=conf_loss_neg.sum()
                loc_all+=loc_loss.sum()
                total_loss+=(conf_loss + self.alpha*loc_loss.sum())/pos[i].sum().float()
  
            total_loss=total_loss/batch_size

            return total_loss
        else:
            print(0)
            return 0

        
        