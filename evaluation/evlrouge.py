import torch
import json
from rouge import Rouge

from maskrcnn_benchmark.structures.bounding_box import BoxList


def rouge(a,b):
    rouge = Rouge()  
    # rouge_score1 = rouge.get_scores(a,b)
    # for r in rouge_score1:
    #     print(r["rouge-1"])
    rouge_score = rouge.get_scores(a,b, avg=True) # a和b里面包含多个句子的时候用
    r1 = rouge_score["rouge-1"]
    r2 = rouge_score["rouge-2"]
    rl =rouge_score["rouge-l"]
    
    return r1, r2, rl

def evl():
    print('evl')

    image_file = json.load(open('./scene-graph-benchmark/datasets/vg/image_data.json'))
    image_to_id = {}
    for i in image_file:
        # print(i.keys())
        imagepath = i['url']
        imageid = i['image_id']
        image_to_id[imagepath.split('/')[-1]] = imageid
    # print(image_file[0])
    print('-----------------------------image to id----------------------------------------')

    qa_info = json.load(open('./question_answers.json'))
    qa_dict = {}
    for i in qa_info:
        # print(i.keys())
        img = i["id"]
        qa = []
        for j in i["qas"]:
            qa.append(j["question"])
        qa_dict[img] = qa
    # print(qa_dict[2317993])
    print('-----------------------------id to qas-------------------------------------')

    detected_origin_path = './checkpoints/vctree-sgdet-attr-exmp/inference/VG_stanford_filtered_with_attribute_test/'
    # detected_origin_result = torch.load(detected_origin_path + 'eval_results.pytorch')
    # groundtruths = detected_origin_result['groundtruths']
    # print('-----------------------------groundtruths---------------------------------------')

    question_result = json.load(open(detected_origin_path + 'output/2_gt.json'))
    questions = question_result['questions']
    print('-----------------------------questions---------------------------------------')



    print(len(questions))
    # print(question_result['questions'][1].keys())

    kong = 0
    myr1,myr2,myrl = 0,0,0
    gtr1,gtr2,gtrl = 0,0,0
    for idx,question in enumerate(questions):
        print(idx)
        myq = question['question']
        # print(myq)
        img_path = question['image_filename']
        img_id = image_to_id[img_path.split('/')[-1]]
        qas = qa_dict[img_id]
        # print(qas)
        if len(qas) == 0:
            kong = kong + 1
            continue
        gtq = qas[0]
        qas.pop(0)
        myqs = []
        gtqs = []
        allqs = []
        for q in qas:
            myqs.append(myq)
            gtqs.append(gtq)
            allqs.append(q)
        if len(allqs) == 0: kong = kong + 1
        else:
            r1, r2, rl = rouge(gtqs,allqs)
            gtr1 = gtr1 + r1['r']
            gtr2 = gtr2 + r2['r']
            gtrl = gtrl + rl['r']
            myqs.append(myq)
            allqs.append(gtq)
            r1, r2, rl = rouge(myqs,allqs)
            myr1 = myr1 + r1['r']
            myr2 = myr2 + r2['r']
            myrl = myrl + rl['r']
            # print(myq)
            # print(gtq)
            # print(allqs)
        print('myr1: ',myr1/(idx+1-kong),'gtr1: ',gtr1/(idx+1-kong))
        print('myr2: ',myr2/(idx+1-kong),'gtr2: ',gtr2/(idx+1-kong))
        print('myrl: ',myrl/(idx+1-kong),'gtrl: ',gtrl/(idx+1-kong))
        # if idx == 5 : break
    
    print()
    print('myr1: ',myr1/(len(questions)-kong),'gtr1: ',gtr1/(len(questions)-kong))
    print('myr2: ',myr2/(len(questions)-kong),'gtr2: ',gtr2/(len(questions)-kong))
    print('myrl: ',myrl/(len(questions)-kong),'gtrl: ',gtrl/(len(questions)-kong))


    return -1

evl()