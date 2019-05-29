import torch
import torchvision
from torchvision import transforms
from PIL import Image
from time import time
import numpy as np

import torch
import torchvision.transforms as transforms

from torch.autograd import Variable

from retinanet_for_export import RetinaNet
# from retinanet import RetinaNet
from encoder import DataEncoder
from PIL import Image, ImageDraw
from glob import glob
from os.path import join,exists,split
import os
import cv2
import time
import numpy as np
print('Loading model..')
model = RetinaNet(1)
keys=torch.load('G:\PycharmProjects\pytorch-retinanet-master\checkpoint\ckpt.pth')
model.load_state_dict(keys['net'])
model.eval()
model.cuda()
model.eval()

# An example input you would normally provide to your model's forward() method.
example = torch.rand(1, 3, 1280, 960).cuda()

# Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
traced_script_module = torch.jit.trace(model, example)
traced_script_module.save("model.pt")


# read image
image = Image.open('IMG_3321.JPG').convert('RGB')

image=image.resize((1280,960))
img=image.copy()
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))
])
print(image.size)
image = transform(image)

# forward
loc_preds= traced_script_module(image.unsqueeze(0).cuda())
loc_preds=loc_preds.argmax()
print(loc_preds)
encoder = DataEncoder()
ref_table = encoder._get_anchor_boxes(torch.Tensor([1280, 960]))

boxes = [ref_table[loc_preds]]
box=boxes[0]
print(boxes)
box[0] = (box[0] - box[2]/2)
box[1] = (box[1] - box[3]/2)
box[2] = (box[2] + box[0])
box[3] = (box[3] + box[1])

print(ref_table[215999])

draw = ImageDraw.Draw(img)

draw.rectangle(list(box), outline='red')
img.show()
