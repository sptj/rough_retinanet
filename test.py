import torch
import torchvision.transforms as transforms

from torch.autograd import Variable

from retinanet_for_train import RetinaNet
from encoder import DataEncoder
from PIL import Image, ImageDraw
from glob import glob
from os.path import join,exists,split
import os
import cv2
print('Loading model..')
net = RetinaNet(1)
keys=torch.load('./checkpoint/ckpt.pth')
net.load_state_dict(keys['net'])
net.eval()

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))
])

print('Loading image..')

def predict1920x1080(img):
    img=img.resize((1920,1080))
    w = h = 1024
    img_left=img.crop((0,0,1023,1023))
    img_right=img.crop((1919-1024,0,1919,1023))
    def predict(img):
        print('Predicting..')
        x = transform(img)
        x = x.unsqueeze(0)
        x = Variable(x, volatile=True)
        loc_preds, cls_preds = net(x)
        print('Decoding..')
        encoder = DataEncoder()
        boxes, labels = encoder.decode(loc_preds.data.squeeze(), cls_preds.data.squeeze(), (w, h))
        draw = ImageDraw.Draw(img)
        for box in boxes:
            draw.rectangle(list(box), outline='red')
        return img
    img_left=predict(img_left)
    img_right=predict(img_right)
    img.paste(img_left,(0,0,1023,1023))
    img.paste(img_right,(1919-1024,0,1919,1023))
    return img
def predict1024x1024(filename,output_dir):
    if not exists(output_dir):
        os.mkdir(output_dir)
    img = Image.open(filename)
    w = h = 1024
    # img=img.resize((2*w,h))

    # def center_crop(img):
    #     img_w,img_h=img.size
    #     (left, upper, right, lower)=(img_w/2-w/2),(img_h/2-h/2),(img_w/2+w/2),(img_h/2+h/2)
    #     # img.crop((left, lower, right, upper))
    #     return img.crop((left, upper, right, lower))
    # img=center_crop(img)
    img = img.resize((w, h))
    # img_left=img.crop((0,0,1023,1023))
    # img_right=img.crop((1919-1024,0,1919,1023))
    def predict(img):
        print('Predicting..')
        w = h = 1024
        x = transform(img)
        x = x.unsqueeze(0)
        x = Variable(x, volatile=True)
        loc_preds, cls_preds = net(x)
        print('Decoding..')
        encoder = DataEncoder()
        boxes, labels = encoder.decode(loc_preds.data.squeeze(), cls_preds.data.squeeze(), (w, h))
        draw = ImageDraw.Draw(img)
        for box in boxes:
            draw.rectangle(list(box), outline='red')
        return img
    img=predict(img)
    img.save(join(output_dir,split(filename)[1]))
def predict_video(video_name,output_dir):
    video_capture=cv2.VideoCapture(video_name)
    status,frame=video_capture.read()
    frame_count=video_capture.get(cv2.CAP_PROP_FRAME_COUNT)
    i=0
    while status:
        i=i+1
        status, frame = video_capture.read()
        img = Image.fromarray(frame)
        img = img.resize((1920, 1080))
        img = predict1920x1080(img)
        img.save(join(output_dir, os.path.split(video_name)[-1].split('.')[0] + '_frame_' + str(i) + '.png'))
        if(i%100==0):

            print('processed',int(i/frame_count*100),'%')
    video_capture.release()





def test(test_data_dir,output_dir):
    suffix='*.mp4'
    test_filenames=glob(join(test_data_dir,suffix))
    from random import shuffle
    shuffle(test_filenames)
    for file_name in test_filenames:
        predict_video(file_name,output_dir)

if __name__ == '__main__':
    import torch as t

    # test(r'D:\drone_image_and_annotation_mixed\test',r'D:\output')
    # test(r'D:\test',r'D:\output')
    # test(r'D:\test_drone_video',r'D:\output')
    predict_video(r'E:\video_save_燕郊\video_save\video20181119_1243.mp4', r'D:\output')