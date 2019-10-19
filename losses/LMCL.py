import torch
import torch.nn as nn

def LMCL(margin,input_dim,output_dim):
    #https://github.com/yule-li/CosFace
    return LMCLLoss(input_dim=input_dim, output_dim=output_dim,margin=margin)
class LMCLLoss(nn.Module):

    def __init__(self, input_dim, output_dim, margin):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.margin = margin
        self.weight = nn.Parameter(torch.FloatTensor(input_dim, output_dim)).cuda()
        self.scale = 64
        self.criterion = nn.CrossEntropyLoss()
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.weight.data.t())

    def forward(self, input, target=None):  #input [batch, embed_size],weight [embed_size, class_number]
        input_norm = input.norm(dim=1)  #[batch,1]  
        weight_norm = self.weight.norm(dim=0)  #[cls_num,1]
        input_normalization = (input.t()/input_norm).t()
        weight_normalization = self.weight/weight_norm
        xw = input_normalization.matmul(weight_normalization)
        num = len(target)
        orig_ind = range(num)
        xw[orig_ind,target] -= self.margin
        xw *= self.scale

        return self.criterion(xw,target) 