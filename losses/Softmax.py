import torch.nn as nn

def Softmax(margin,input_dim,output_dim):
    return SoftmaxLoss(margin=margin, input_dim=input_dim, output_dim=output_dim)
class SoftmaxLoss(nn.Module):

    def __init__(self, input_dim, output_dim, margin):
        super(SoftmaxLoss, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.criterion = nn.CrossEntropyLoss()
        self.logits = nn.Linear(in_features=self.input_dim, out_features=self.output_dim).cuda()
    def forward(self, input, target=None):
        return self.criterion(self.logits(input),target)