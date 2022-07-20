import torch
import json
from PIL import Image, ImageDraw
import random

def hevldata():

    print('hevldata')

    image_file = json.load(open('./scene-graph-benchmark/datasets/vg/image_data.json'))
    image_to_id = {}
    for i in image_file:
        # print(i.keys())
        imagepath = i['url']
        imageid = i['image_id']
        image_to_id[imagepath.split('/')[-1]] = imageid
    print('-----------------------------image to id----------------------------------------')

    qa_info = json.load(open('./question_answers.json'))
    qa_dict = {}
    for i in qa_info:
        # print(i.keys())
        img = i["id"]
        qa = []
        for j in i["qas"][:3]:
            qa.append(j["question"]+ ' ' + j["answer"])
        qa_dict[img] = qa
    # print(qa_dict[2317993])
    # return -1
    print('-----------------------------id to qas-------------------------------------')
    
    detected_origin_path = './'
    # detected_origin_result = torch.load(detected_origin_path + 'eval_results.pytorch')
    # groundtruths = detected_origin_result['groundtruths']
    # print('-----------------------------groundtruths---------------------------------------')
    detected_info = json.load(open(detected_origin_path + 'visual_info.json'))
    print('-----------------------------visual info----------------------------------------')

    q_list = []
    for info in detected_info:
        q_list.append([])

    question_result = json.load(open(detected_origin_path + 'output/0_1.json'))
    questions0 = question_result['questions']
    for q in questions0:
        imgid = q['image_index']
        q_list[imgid].append(q['question']+ ' ' + str(q['answer']))
    print('-----------------------------questions0---------------------------------------')

    question_result = json.load(open(detected_origin_path + 'output/1_test1.json'))
    questions1 = question_result['questions']
    for q in questions1:
        imgid = q['image_index']
        q_list[imgid].append(q['question']+ ' ' + q['answer'])
    print('-----------------------------questions1---------------------------------------')

    question_result = json.load(open(detected_origin_path + 'output/2_test1.json'))
    questions2 = question_result['questions']
    for q in questions2:
        imgid = q['image_index']
        q_list[imgid].append(q['question']+ ' ' + q['answer'])
    print('-----------------------------questions2---------------------------------------')

    datavg = []
    dataai = []
    for it,info in enumerate(detected_info[400:600]):
        i = it+400
        if len(q_list[i])>=3:
            img_path = info['img_file']
            tu = Image.open(img_path)
            tu.save('imgs/'+img_path.split('/')[-1])

            img_id = image_to_id[img_path.split('/')[-1]]
            qas = qa_dict[img_id]
            one = {}
            one["q"] = qas
            one["i"] = img_path.split('/')[-1]
            one["f"] = "vg"
            datavg.append(one)

            one = {}
            one["q"] = q_list[i][-3:]
            one["i"] = img_path.split('/')[-1]
            one["f"] = "ai"
            dataai.append(one)

        # if i==3:break
    
    human1 = []
    human2 = []
    for idx in range(0,len(dataai)):
        if idx < len(dataai)/2:
            human1.append(dataai[idx])
            human2.append(datavg[idx])
        else:
            human2.append(dataai[idx])
            human1.append(datavg[idx])
    random.shuffle(human1)
    random.shuffle(human2)
    d1 = {
        "info":"400/3",
        "data":human1
    }
    d2 = {
        "info":"400/4",
        "data":human2
    }
    with open('data3.json', 'w') as outfile: 
            json.dump(d1, outfile)
    with open('data4.json', 'w') as outfile: 
            json.dump(d2, outfile)
        

    return -1

hevldata()