from ast import IsNot
import torch
from torch import nn
from models.S_Former import spatial_transformer
from models.T_Former import temporal_transformer
from einops import rearrange, repeat
import torch.nn.functional as F
import scipy.stats
def js_div(p_output, q_output, get_softmax=False):
    """
    Function that measures JS divergence between target and output logits:
    """

    if get_softmax:
        p_output = F.softmax(p_output)
        q_output = F.softmax(q_output)
    log_mean_output = ((p_output + q_output )/2).log()
    return (F.kl_div(log_mean_output, p_output,reduce=None,reduction='none') + F.kl_div(log_mean_output, q_output,reduce=None,reduction='none'))/2

class GenerateModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.s_former = spatial_transformer()
        pretrained_checkpoint_path = './checkpoint/aff_pre_spatial.pth'
        pre_trained_dict = torch.load(pretrained_checkpoint_path)['state_dict']
        for key in list(pre_trained_dict.keys()):
            pre_trained_dict[key.replace('module.s_former.','')] = pre_trained_dict.pop(key)
        self.s_former.load_state_dict(pre_trained_dict)

        self.t_former = temporal_transformer()
        self.softmax=nn.Softmax(dim=1)
        self.CE = nn.CrossEntropyLoss(reduce=None,reduction='none')
        self.fc_spatial_end = nn.Linear(512,7)
        self.fc_spatial_middle=nn.Linear(256,7)
        self.fc = nn.Linear(512, 7)


    def forward(self, x):
        x,middle_output= self.s_former(x)
        spatial_output_end=x
        spatial_output_end=self.fc_spatial_end(spatial_output_end)   #(512,7)
        spatial_output_middle=self.fc_spatial_middle(middle_output)   #(512,7)

        y=torch.ones((x.size()[0],7))/7        
        y=y.cuda()
        ce_end=self.CE(spatial_output_end,y).unsqueeze(1)   
        ce_middle=self.CE(spatial_output_middle,y).unsqueeze(1)
        x=x*(ce_end+ce_middle)/2
        feature = self.t_former(x)
        x = self.fc(feature)
        return x,spatial_output_end,spatial_output_middle
