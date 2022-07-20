import torch
import json
from PIL import Image, ImageDraw
import random

def reform():

    print('reform')

    image_file = json.load(open('./datasets/vg/image_data.json'))
    image_to_id = {}
    for i in image_file:
        # print(i.keys())
        imagepath = i['url']
        imageid = i['image_id']
        image_to_id[imagepath.split('/')[-1]] = imageid
    print('-----------------------------image to id----------------------------------------')

    
    detected_origin_path = './'
    # detected_origin_result = torch.load(detected_origin_path + 'eval_results.pytorch')
    # groundtruths = detected_origin_result['groundtruths']
    # print('-----------------------------groundtruths---------------------------------------')
    detected_info = json.load(open(detected_origin_path + 'visual_info.json'))
    print('-----------------------------visual info----------------------------------------')

    tj = {
        '0':0,
        '1':0,
        '2':0
    }
    questionlist = []
    

    for i in range(1,2):
        question_result = json.load(open(detected_origin_path + 'output/0_val.json'))
        questions = question_result['questions']
        for q in questions:
            ques = {}
            ques['split'] = 'val'
            ques['hop'] = 0

            newname = str(q['image_index'])
            que = 6-len(newname)
            newname = 'AI_val_' + '0'*que + newname + '.jpg'

            ques['image_filename'] = newname
            ques['image_index'] = q['image_index']
            ques['question'] = q['question']
            ques['answer'] = q['answer']
            ques['type'] = q['program'][-1]['type']

            tj['0'] = tj['0'] + 1
            if ques['type'] in tj.keys():
                tj[ques['type']] = tj[ques['type']] + 1
            else:
                tj[ques['type']] = 1

            ques['question_idx'] = len(questionlist)
            questionlist.append(ques)
            img_path = q['image_filename']
            tu = Image.open(img_path)
            tu.save('./AIVQA/images/val/'+ newname)
            print(ques['question_idx'],ques['image_index'])
            # break

    
    print('-----------------------------questions0---------------------------------------')

    for i in range(1,2):
        question_result = json.load(open(detected_origin_path + 'output/1_val.json'))
        questions = question_result['questions']
        for q in questions:
            ques = {}
            ques['split'] = 'val'
            ques['hop'] = 1

            newname = str(q['image_index'])
            que = 6-len(newname)
            newname = 'AI_val_' + '0'*que + newname + '.jpg'

            ques['image_filename'] = newname
            ques['image_index'] = q['image_index']
            ques['question'] = q['question']
            ques['answer'] = q['answer']
            ques['type'] = q['program'][-1]['type']

            tj['1'] = tj['1'] + 1
            if ques['type'] in tj.keys():
                tj[ques['type']] = tj[ques['type']] + 1
            else:
                tj[ques['type']] = 1

            ques['question_idx'] = len(questionlist)
            questionlist.append(ques)
            img_path = q['image_filename']
            tu = Image.open(img_path)
            tu.save('./AIVQA/images/val/'+ newname)
            print(ques['question_idx'],ques['image_index'])
            # break

    
    # print('-----------------------------questions1---------------------------------------')

    for i in range(1,3):
        question_result = json.load(open(detected_origin_path + 'output/2_val.json'))
        questions = question_result['questions']
        for q in questions:
            ques = {}
            ques['split'] = 'val'
            ques['hop'] = 2

            newname = str(q['image_index'])
            que = 6-len(newname)
            newname = 'AI_val_' + '0'*que + newname + '.jpg'

            ques['image_filename'] = newname
            ques['image_index'] = q['image_index']
            ques['question'] = q['question']
            ques['answer'] = q['answer']
            ques['type'] = q['program'][-1]['type']

            tj['2'] = tj['2'] + 1
            if ques['type'] in tj.keys():
                tj[ques['type']] = tj[ques['type']] + 1
            else:
                tj[ques['type']] = 1

            ques['question_idx'] = len(questionlist)
            questionlist.append(ques)
            img_path = q['image_filename']
            tu = Image.open(img_path)
            tu.save('./AIVQA/images/val/'+ newname)
            print(ques['question_idx'],ques['image_index'])
            # break
    
    # print('-----------------------------questions2---------------------------------------')

    # print(questionlist)

    data = {
        "info":{
            "split":"val",
            "version":"v1",
            "data":"AIVQA"
        },
        "questions":questionlist
    }
    
    with open('./AIVQA/questions/AI_val_questions.json', 'w') as outfile: 
            json.dump(data, outfile)
    print('down')
    print(tj)
        

    return -1

reform()