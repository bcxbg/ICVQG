import torch
import json
from PIL import Image, ImageDraw
import random

def buimg():

    print('buimg')

    image_file = json.load(open('./scene-graph-benchmark/datasets/vg/image_data.json'))
    image_to_id = {}
    for i in image_file:
        # print(i.keys())
        imagepath = i['url']
        imageid = i['image_id']
        image_to_id[imagepath.split('/')[-1]] = imageid
    print('-----------------------------image to id----------------------------------------')

    
    detected_origin_path = './'
    detected_info = json.load(open(detected_origin_path + 'visual_info.json'))
    print('-----------------------------val visual info----------------------------------------')

    for idx,info in enumerate(detected_info):
        print(idx)
        newname = str(idx)
        que = 6-len(newname)
        newname = 'AI_val_' + '0'*que + newname + '.jpg'

        img_path = info['img_file']
        tu = Image.open(img_path)
        tu.save('./AIVQA/images/val/'+ newname)
    

    detected_origin_path = './'
    detected_info = json.load(open(detected_origin_path + 'visual_info.json'))
    print('-----------------------------test visual info----------------------------------------')

    for idx,info in enumerate(detected_info):
        print(idx)
        newname = str(idx)
        que = 6-len(newname)
        newname = 'AI_test_' + '0'*que + newname + '.jpg'

        img_path = info['img_file']
        tu = Image.open(img_path)
        tu.save('./AIVQA/images/test/'+ newname)
    

    detected_origin_path = './'
    detected_info = json.load(open(detected_origin_path + 'visual_info.json'))
    print('-----------------------------val visual info----------------------------------------')

    for idx,info in enumerate(detected_info):
        print(idx)
        newname = str(idx)
        que = 6-len(newname)
        newname = 'AI_train_' + '0'*que + newname + '.jpg'

        img_path = info['img_file']
        tu = Image.open(img_path)
        tu.save('./AIVQA/images/train/'+ newname)
    



    

    
        

    return -1

buimg()