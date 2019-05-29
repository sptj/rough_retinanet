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
import time
import numpy as np
print('Loading model..')
net = RetinaNet(1)
keys=torch.load('G:\PycharmProjects\pytorch-retinanet-resnet50\checkpoint\ckpt-from_zero.pth')
net.load_state_dict(keys['net'])
net.eval()
net.cuda(0)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485,0.456,0.406), (0.229,0.224,0.225))
])

print('Loading image..')
time_cost=[]
def predict1920x1080(img):
    img=img.resize((1920,1080))
    w = h = 1024
    img_left=img.crop((0,0,1023,1023))
    img_right=img.crop((1024,0,2047,1023))
    def predict(img):
        print('Predicting..')

        x = transform(img)
        start_time=time.time()
        x = x.unsqueeze(0)
        x = Variable(x.cuda(0), volatile=True)
        # x.cuda()
        loc_preds, cls_preds = net(x)
        end_time=time.time()
        print('Decoding..')
        encoder = DataEncoder()
        boxes, labels = encoder.decode(loc_preds.data.squeeze().cpu(), cls_preds.data.squeeze().cpu(), (w, h))
        print('infer time:   \t',end_time - start_time)
        print('decoder time: \t',time.time()-end_time)

        draw = ImageDraw.Draw(img)
        for box in boxes:
            draw.rectangle(list(box), outline='red')
        return img
    img_left=predict(img_left)
    img_right=predict(img_right)
    img.paste(img_left,(0,0,1023,1023))
    img.paste(img_right,(1024,0,2047,1023))
    return img
def predict1024x1024(filename,output_dir):
    if not exists(output_dir):
        os.mkdir(output_dir)
    img = Image.open(filename)
    w = h = 1024
    img = img.resize((w, h))
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
    fourcc_specific = {
        '.mp4': 'MJPG',
        '.avi': 'XVID',
        '.ogv': 'THEO',
        '.flv': 'FLV1',
        '.wmv': 'MJPG',
        '.mkv': '3IVX',
    }
    dst_filename = join(output_dir,'1.wmv')
    (_, file_ext) = os.path.splitext(dst_filename)
    file_ext = file_ext.lower()
    if file_ext not in fourcc_specific:
        print('dist format not support')
        exit(0)
    fourcc_type = fourcc_specific[file_ext]
    #video_writer = cv2.VideoWriter(dst_filename, cv2.VideoWriter_fourcc(*fourcc_type), (50), (1920, 1080))
    while status:
        i=i+1
        status, frame = video_capture.read()
        frame=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        img = img.resize((1920, 1080))

        img = predict1920x1080(img)
        # video_writer.write(cv2.cvtColor(numpy.asarray(img),cv2.COLOR_RGB2BGR))
        img.save(join(output_dir, os.path.split(video_name)[-1].split('.')[0] + '_frame_' + str(i) + '.png'))
        # cv2.imshow('',cv2.cvtColor(numpy.asarray(img),cv2.COLOR_RGB2BGR))
        # cv2.waitKey(1)
        if(i%100==0):

            print('processed',int(i/frame_count*100),'%')
    video_capture.release()


global end_time
end_time=0
def predict_1080P(img):
    def predict(img):
        global end_time
        print('Predicting..')
        w = 1920
        h = 1080
        x = transform(img)
        x = x.unsqueeze(0)
        x = Variable(x.cuda(0), volatile=True)
        start_time=time.time()
        loc_preds, cls_preds = net(x)
        print("proid time",time.time()-end_time)
        end_time=time.time()
        time_cost=end_time-start_time
        print("time_cost",time_cost)
        print('Decoding..')
        encoder = DataEncoder()
        n1=time.time()
        # boxes, labels = encoder.decode(loc_preds.data.squeeze().cpu(), cls_preds.data.squeeze().cpu(), (w, h))
        ref_table = encoder._get_anchor_boxes(torch.Tensor([1920,1080]))
        cls_preds = cls_preds.flatten()
        idx=cls_preds.argmax()
        print(idx)
        print(cls_preds[idx])
        boxes=[ref_table[idx]]
        n2=time.time()
        print("useless",n2-n1)
        draw = ImageDraw.Draw(img)
        for box in boxes:
            draw.rectangle(list(box), outline='red')
        return img
    img=predict(img)
    return img

def test(test_data_dir,output_dir):
    suffix='*.mp4'
    test_filenames=glob(join(test_data_dir,suffix))
    from random import shuffle
    shuffle(test_filenames)
    for file_name in test_filenames:
        predict_video(file_name,output_dir)

if __name__ == '__main__':

    # test(r'D:\drone_image_and_annotation_mixed\test',r'D:\output')
    # test(r'D:\test',r'D:\output')
    # test(r'D:\test_drone_video',r'D:\output')
    predict_video(r'E:\static camera\video2019115_16284.mp4', r'D:\output')
    # cap = cv2.VideoCapture("rtsp://admin:jishukaifa3432@192.168.1.64:554/Streaming/Channels/101?transportmode=unicast")
    # i=0
    # while (cap.isOpened()):
    #     i+=1
    #     ret, frame = cap.read()
    #     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #     img = Image.fromarray(frame)
    #     # img = img.resize((1920, 1080))
    #
    #     img = predict_1080P(img)
    #     # video_writer.write(cv2.cvtColor(numpy.asarray(img),cv2.COLOR_RGB2BGR))
    #     img.save(join("D:/output", '_frame_' + str(i) + '.png'))
    #     cv2.imshow('',cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR))
    #     cv2.waitKey(1)
    #
    #
