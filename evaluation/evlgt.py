import torch
import json

# from maskrcnn_benchmark.structures.bounding_box import BoxList


import spacy
from spacy_lookup import Entity
entity_dict = {"label": ['airplane', 'animal', 'arm', 'bag', 'banana', 'basket', 'beach', 'bear', 'bed', 'bench', 'bike', 'bird', 'board', 'boat', 'book', 'boot', 'bottle', 'bowl', 'box', 'boy', 'branch', 'building', 'bus', 'cabinet', 'cap', 'car', 'cat', 'chair', 'child', 'clock', 'coat', 'counter', 'cow', 'cup', 'curtain', 'desk', 'dog', 'door', 'drawer', 'ear', 'elephant', 'engine', 'eye', 'face', 'fence', 'finger', 'flag', 'flower', 'food', 'fork', 'fruit', 'giraffe', 'girl', 'glass', 'glove', 'guy', 'hair', 'hand', 'handle', 'hat', 'head', 'helmet', 'hill', 'horse', 'house', 'jacket', 'jean', 'kid', 'kite', 'lady', 'lamp', 'laptop', 'leaf', 'leg', 'letter', 'light', 'logo', 'man', 'men', 'motorcycle', 'mountain', 'mouth', 'neck', 'nose', 'number', 'orange', 'pant', 'paper', 'paw', 'people', 'person', 'phone', 'pillow', 'pizza', 'plane', 'plant', 'plate', 'player', 'pole', 'post', 'pot', 'racket', 'railing', 'rock', 'roof', 'room', 'screen', 'seat', 'sheep', 'shelf', 'shirt', 'shoe', 'short', 'sidewalk', 'sign', 'sink', 'skateboard', 'ski', 'skier', 'sneaker', 'snow', 'sock', 'stand', 'street', 'surfboard', 'table', 'tail', 'tie', 'tile', 'tire', 'toilet', 'towel', 'tower', 'track', 'train', 'tree', 'truck', 'trunk', 'umbrella', 'vase', 'vegetable', 'vehicle', 'wave', 'wheel', 'window', 'windshield', 'wing', 'wire', 'woman', 'zebra']
                ,"rel":['above', 'across', 'against', 'along', 'and', 'at', 'attached to', 'behind', 'belonging to', 'between', 'carrying', 'covered in', 'covering', 'eating', 'flying in', 'for', 'from', 'growing on', 'hanging from', 'has', 'holding', 'in', 'in front of', 'laying on', 'looking at', 'lying on', 'made of', 'mounted on', 'near', 'of', 'on', 'on back of', 'over', 'painted on', 'parked on', 'part of', 'playing', 'riding', 'says', 'sitting on', 'standing on', 'to', 'under', 'using', 'walking in', 'walking on', 'watching', 'wearing', 'wears', 'with']
                ,"att":['white', 'black', 'blue', 'green', 'red', 'brown', 'yellow', 'small', 'large', 'wooden', 'silver', 'orange', 'grey', 'tall', 'long', 'dark', 'pink', 'standing', 'round', 'tan', 'glass', 'here', 'wood', 'open', 'purple', 'short', 'plastic', 'parked', 'sitting', 'walking', 'striped', 'brick', 'young', 'gold', 'old', 'hanging', 'empty', 'on', 'bright', 'concrete', 'cloudy', 'colorful', 'one', 'beige', 'bare', 'wet', 'light', 'square', 'closed', 'stone', 'shiny', 'thin', 'dirty', 'flying', 'smiling', 'painted', 'thick', 'part', 'sliced', 'playing', 'tennis', 'calm', 'leather', 'distant', 'rectangular', 'looking', 'grassy', 'dry', 'cement', 'leafy', 'wearing', 'tiled', "man", 'baseball', 'cooked', 'pictured', 'curved', 'decorative', 'dead', 'eating', 'paper', 'paved', 'fluffy', 'lit', 'back', 'framed', 'plaid', 'dirt', 'watching', 'colored', 'stuffed', 'clean', 'the picture', 'steel', 'stacked', 'covered', 'full', 'three', 'street', 'flat', 'baby', 'beautiful', 'ceramic', 'present', 'grazing', 'sandy', 'golden', 'blurry', 'side', 'chocolate', 'wide', 'growing', 'chrome', 'cut', 'bent', 'train', 'holding', 'water', 'up', 'arched', 'metallic', 'spotted', 'folded', 'electrical', 'pointy', 'running', 'leafless', 'electric', 'background', 'rusty', 'furry', 'traffic', 'ripe', 'behind', 'laying', 'rocky', 'tiny', 'down', 'fresh', 'floral', 'stainless steel', 'high', 'surfing', 'close', 'off', 'leaning', 'moving', 'multicolored', "woman", 'pair', 'huge', 'some', 'background', 'chain link', 'checkered', 'top', 'tree', 'broken', 'maroon', 'iron', 'worn', 'patterned', 'ski', 'overcast', 'waiting', 'rubber', 'riding', 'skinny', 'grass', 'porcelain', 'adult', 'wire', 'cloudless', 'curly', 'cardboard', 'jumping', 'tile', 'pointed', 'blond', 'cream', 'four', 'male', 'smooth', 'hazy', 'computer', 'older', 'pine', 'raised', 'many', 'bald', 'covered', 'skateboarding', 'narrow', 'reflective', 'rear', 'khaki', 'extended', 'roman', 'american']
               }
nlp = spacy.load('en_core_web_sm')  
nlp.remove_pipe('ner')
edict_keys = entity_dict.keys()
for key, values in entity_dict.items():
    entity = Entity(keywords_list=values, label=key)
    nlp.add_pipe(entity, name=key)


def evl():
    print('evl')

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
    
    vocab_file = json.load(open('./datasets/vg/VG-SGG-dicts-with-attri.json'))
    idx2label = vocab_file['idx_to_label']
    # label2idx = vocab_file['label_to_idx']
    idx2rel = vocab_file['idx_to_predicate']
    idx2att = vocab_file['idx_to_attribute']
    print('-----------------------------id & lable & rel & att-----------------------------------------')

    detected_origin_path = './'
    detected_origin_result = torch.load(detected_origin_path + 'eval_results.pytorch')
    groundtruths = detected_origin_result['groundtruths']
    print('-----------------------------groundtruths---------------------------------------')
    detected_info = json.load(open(detected_origin_path + 'visual_info.json'))
    print('-----------------------------visual info----------------------------------------')

    # question_result = json.load(open(detected_origin_path + 'output/0_2.json'))
    # questions = question_result['questions']
    # print('-----------------------------questions---------------------------------------')

    # print(len(questions))
    # print(question_result['questions'][1].keys())

    # match = []
    kong = 0
    suma = 0
    sumr = 0
    for idx,gt in enumerate(groundtruths):
        print(idx)
        tset = set()
        labels = gt.get_field('labels').tolist()
        for l in labels:
            tset.add(idx2label[str(l)])
        att = gt.get_field('attributes').tolist()
        for l in att:
            for a in l:
                if not(a == 0):
                    tset.add(idx2att[str(a)])
        relt = gt.get_field('relation_tuple').tolist()
        for r in relt:
            tset.add(idx2rel[str(r[2])])
        # match.append(tset)

        img_path = detected_info[idx]['img_file']
        img_id = image_to_id[img_path.split('/')[-1]]
        qas = qa_dict[img_id]
        print(qas)

        for q in qas:
            words = 0
            ele = 0
            trele = 0
            doc = nlp(q)
            for token in doc:
                # print(token.text,token._.is_entity)
                words = words + 1
                if token._.is_entity:
                    ele = ele + 1
                    if token.text in tset:
                        # print('match')
                        trele = trele + 1
            # print(trele,ele,words)
            if ele == 0: ai = 0
            else: ai = trele/ele
            suma = suma + ai
            if words == 0: ri = 0
            else: ri = ele/words
            sumr = sumr + ri
        if len(qas) == 0: kong = kong + 1
        print(suma/((idx+1-kong)*3),sumr/((idx+1-kong)*3))



        # print(tset)
        print()
        # if idx == 3:break
    
    
    print()
    print(suma,sumr,kong)
    print(suma/((len(groundtruths)-kong)*3),sumr/((len(groundtruths)-kong)*3))





    return -1

evl()