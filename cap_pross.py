import gc
from io import BytesIO
import torch
import torchvision.transforms as transforms
from torch.autograd import Variable
from retinanet_for_train import RetinaNet
from encoder import DataEncoder
from PIL import Image
global transform, net

def NVIDA_INIT():
    global transform, net
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    net = RetinaNet(1)
    keys = torch.load('G:\PycharmProjects\pytorch-retinanet-master\checkpoint\ckpt.pth')
    net.load_state_dict(keys['net'])
    net.eval()
    net.cuda(0)
    encoder = DataEncoder()
    
    print("NVIDA_INIT success")
    return (0, 0.0, 0.0, 0.0, 0.0)


def pic_process(width, height, data_list):
    global transform, net
    img = Image.open(BytesIO(data_list))
    x = transform(img)
    x = x.unsqueeze(0)
    x = Variable(x.cuda(0), volatile=True)
    loc_preds, cls_preds = net(x)
    encoder = DataEncoder()
    boxes, labels = encoder.decode(loc_preds.data.squeeze().cpu(), cls_preds.data.squeeze().cpu(), (width, height))
    if (len(boxes)!=0):
        result = 5 * [1]
        result[1:5] = boxes[0]
        del data_list
        del img
        gc.collect()
        return tuple(result)
    else:
        del data_list
        del img
        gc.collect()
        return (0, 0.0, 0.0, 0.0, 0.0)

# pic_process(1920,1080)
