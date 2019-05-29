import torch
import torch.nn as nn

from fpn_resnet50 import FPN50


class RetinaNet(nn.Module):
    num_anchors = 9
    
    def __init__(self, num_classes):
        super(RetinaNet, self).__init__()
        self.fpn = FPN50()
        self.num_classes = num_classes
        self.phantom_head = self._make_head(self.num_anchors)
        self.mavic_head=self._make_head(self.num_anchors)
        self.initialize()
    def forward(self, x):
        fms = self.fpn(x)
        phantom_preds = []
        mavic_preds = []
        for fm in fms:
            phantom_pred = self.phantom_head(fm)
            mavic_pred = self.mavic_head(fm)
            phantom_pred = phantom_pred.permute(0,2,3,1).contiguous().view(x.size(0),-1,1)
            mavic_pred = mavic_pred.permute(0,2,3,1).contiguous().view(x.size(0),-1,1)
            phantom_preds.append(phantom_pred)
            mavic_preds.append(mavic_pred)
        return torch.cat(phantom_preds,1)#, torch.cat(mavic_preds,1)
    def initialize(self):
        """Initializes the model parameters"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
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
    import termcolor
    net = RetinaNet(1)
    tar_net_dict=net.state_dict()
    src_net_dict=torch.load(r'G:\PycharmProjects\pytorch-retinanet-resnet50\checkpoint\ckpt-best-one.pth')['net']
    for net_param_name,net_param_value in src_net_dict.items():
        if net_param_name.startswith('cls_head'):
            net_param_name=net_param_name.replace('cls_head','phantom_head')
        if net_param_name in tar_net_dict:
            pos_match=net_param_name+'in target net dict'
            print(termcolor.colored(pos_match,'green'))
            tar_net_dict[net_param_name]=net_param_value
        else:
            neg_match=net_param_name+'in target net dict'
            print(termcolor.colored(neg_match,'red'))
    torch.save(tar_net_dict,'tar_net_dict.pt')



    #net.cuda(1)
    # loc_preds, cls_preds = net(Variable(torch.ones(1,3,1920,1080,1)).cuda(1))
    # print( cls_preds.shape,loc_preds.shape,)
    # print( cls_preds[0][0][0],loc_preds[0,0,0],)
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