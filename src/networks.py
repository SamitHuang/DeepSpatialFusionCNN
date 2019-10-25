import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
from torchvision import  models
from options import ModelOptions

args = ModelOptions().parse(verbose=0)
net_type =args.net_type

# this is good
class BaseNetwork(nn.Module):
    def __init__(self, name, channels=1):
        super(BaseNetwork, self).__init__()
        self._name = name
        self._channels = channels

    def name(self):
        return self._name
    # good way to initialize weight
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform(m.weight, gain=np.sqrt(2))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

def softmax(input, dim=1):
    transposed_input = input.transpose(dim, len(input.size()) - 1)
    softmaxed_output = F.softmax(transposed_input.contiguous().view(-1, transposed_input.size(-1)))
    return softmaxed_output.view(*transposed_input.size()).transpose(dim, len(input.size()) - 1)

class PatchWiseNetwork(BaseNetwork):
    def __init__(self, channels=1):
        super(PatchWiseNetwork, self).__init__('pw' + str(channels), channels)
        if(net_type=="resnet50"):
           tl_model=models.resnet50(pretrained=True) #True)
        else:
            tl_model=models.resnet18(pretrained=True) #True)

        #for param in tl_model.parameters():
    	#    param.requires_grad = False
        #num_ftrs = tl_model.fc.in_features
        self.features = nn.Sequential(*list(tl_model.children())[:-2])
        self.pool = nn.AvgPool2d(7,stride=1) 
        self.classifier = nn.Sequential(
                nn.Linear(512*10*10, 4),)

        self.initialize_weights()
    
    def forward(self, x):
        #print(x.data.shape)
        x = self.features(x)
        x = self.pool(x)
        #print(x.data.shape)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        x = F.log_softmax(x, dim=1)
        return x

class ImageWiseNetwork(BaseNetwork):
    def __init__(self, channels=1):
        super(ImageWiseNetwork, self).__init__('iw' + str(channels), channels)
        if(args.patches_overlap):
            patches_num=35 #35 if overlap patches ,12 if non-overlap
        else:
            patches_num=12
        if("2018" in args.dataset_path): # layer width=256 for 2018 dataset, 196 for 2015 datast
            self.classifier = nn.Sequential(
                # shallow
                nn.Dropout(p=0.4),
                nn.Linear(patches_num * 4, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.4),
                nn.Linear(256,256 ),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.4),
                nn.Linear(256, 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, 4),
            )
        else:
             self.classifier = nn.Sequential(
                #deep 
                nn.Dropout(),
                nn.Linear(patches_num * 4, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(256,256 ),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(256, 256),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(256, 128),
                nn.ReLU(inplace=True),
                nn.Linear(128, 4),
            )


        self.initialize_weights()

    def forward(self, x):
        #x = self.features(x)
        x = x.view(x.size(0), -1)
        #print(x.data.shape)
        x = self.classifier(x)
        x = F.log_softmax(x, dim=1)
        return x
