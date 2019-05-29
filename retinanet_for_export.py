import torch
import torch.nn as nn

from fpn_resnet34 import FPN34
from torch.autograd import Variable


class RetinaNet(nn.Module):
    num_anchors = 9
    
    def __init__(self, num_classes):
        super(RetinaNet, self).__init__()
        self.fpn = FPN34()
        self.num_classes = num_classes
        self.loc_head = self._make_head(self.num_anchors*4)
        self.cls_head = self._make_head(self.num_anchors*self.num_classes)

    def forward(self, x):
        fms = self.fpn(x)
        loc_preds = []
        cls_preds = []
        for fm in fms:
            loc_pred = self.loc_head(fm)
            cls_pred = self.cls_head(fm)
            loc_pred = loc_pred.permute(0,2,3,1).contiguous().view(x.size(0),-1,4)                 # [N, 9*4,H,W] -> [N,H,W, 9*4] -> [N,H*W*9, 4]
            cls_pred = cls_pred.permute(0,2,3,1).contiguous().view(x.size(0),-1,self.num_classes)  # [N,9*20,H,W] -> [N,H,W,9*20] -> [N,H*W*9,20]
            loc_preds.append(loc_pred)
            cls_preds.append(cls_pred)

        return torch.cat(cls_preds,1)

    def _make_head(self, out_planes):
        layers = []
        for _ in range(4):
            layers.append(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1))
            layers.append(nn.ReLU(True))
        layers.append(nn.Conv2d(256, out_planes, kernel_size=3, stride=1, padding=1))
        return nn.Sequential(*layers)

    def freeze_bn(self):
        '''Freeze BatchNorm layers.'''
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()

def test():
    net = RetinaNet(1)
    net.load_state_dict(torch.load(r'C:\Users\sptj\PycharmProjects\pytorch-retinanet-master\checkpoint\ckpt-ulti.pth')['net'])
    net.cuda(1)
    loc_preds, cls_preds = net(Variable(torch.ones(1,3,1920,1080,1024)).cuda(1))
    print( cls_preds.shape,loc_preds.shape,)
    print( cls_preds[0][0][0],loc_preds[0,0,0],)
    # print(loc_preds.size())
    # print(cls_preds.size())
    # loc_grads = Variable(torch.randn(loc_preds.size()))
    # cls_grads = Variable(torch.randn(cls_preds.size()))
    # loc_preds.backward(loc_grads)
    # cls_preds.backward(cls_grads)

# test()
if __name__ == '__main__':
    # net=FPN50()
    # print(net)
    test()