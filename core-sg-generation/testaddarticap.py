print('test')
# a = {(1,2):set([10,20]),(3,4):set([30,40])}
# for i in a.keys():
#     print(len(a[i]))
# a = [0]*4
# print(a)


import torch
import json

# from maskrcnn_benchmark.structures.bounding_box import BoxList

from sentence_transformers import SentenceTransformer, util
bc = SentenceTransformer('distilbert-base-nli-mean-tokens')

import spacy
from spacy_lookup import Entity
entity_dict = {"label": ['airplane', 'animal', 'arm', 'bag', 'banana', 'basket', 'beach', 'bear', 'bed', 'bench', 'bike', 'bird', 'board', 'boat', 'book', 'boot', 'bottle', 'bowl', 'box', 'boy', 'branch', 'building', 'bus', 'cabinet', 'cap', 'car', 'cat', 'chair', 'child', 'clock', 'coat', 'counter', 'cow', 'cup', 'curtain', 'desk', 'dog', 'door', 'drawer', 'ear', 'elephant', 'engine', 'eye', 'face', 'fence', 'finger', 'flag', 'flower', 'food', 'fork', 'fruit', 'giraffe', 'girl', 'glass', 'glove', 'guy', 'hair', 'hand', 'handle', 'hat', 'head', 'helmet', 'hill', 'horse', 'house', 'jacket', 'jean', 'kid', 'kite', 'lady', 'lamp', 'laptop', 'leaf', 'leg', 'letter', 'light', 'logo', 'man', 'men', 'motorcycle', 'mountain', 'mouth', 'neck', 'nose', 'number', 'orange', 'pant', 'paper', 'paw', 'people', 'person', 'phone', 'pillow', 'pizza', 'plane', 'plant', 'plate', 'player', 'pole', 'post', 'pot', 'racket', 'railing', 'rock', 'roof', 'room', 'screen', 'seat', 'sheep', 'shelf', 'shirt', 'shoe', 'short', 'sidewalk', 'sign', 'sink', 'skateboard', 'ski', 'skier', 'sneaker', 'snow', 'sock', 'stand', 'street', 'surfboard', 'table', 'tail', 'tie', 'tile', 'tire', 'toilet', 'towel', 'tower', 'track', 'train', 'tree', 'truck', 'trunk', 'umbrella', 'vase', 'vegetable', 'vehicle', 'wave', 'wheel', 'window', 'windshield', 'wing', 'wire', 'woman', 'zebra']
                ,"rel":['above', 'across', 'against', 'along', 'and', 'at', 'attached to', 'behind', 'belonging to', 'between', 'carrying', 'covered in', 'covering', 'eating', 'flying in', 'for', 'from', 'growing on', 'hanging from', 'has', 'holding', 'in', 'in front of', 'laying on', 'looking at', 'lying on', 'made of', 'mounted on', 'near', 'of', 'on', 'on back of', 'over', 'painted on', 'parked on', 'part of', 'playing', 'riding', 'says', 'sitting on', 'standing on', 'to', 'under', 'using', 'walking in', 'walking on', 'watching', 'wearing', 'wears', 'with']
               }
nlp = spacy.load('en_core_web_sm')  
nlp.remove_pipe('ner')
edict_keys = entity_dict.keys()
for key, values in entity_dict.items():
    entity = Entity(keywords_list=values, label=key)
    nlp.add_pipe(entity, name=key)

# import pandas as pd


def solve():
    print('addcap')

    

    vocab_file = json.load(open('./datasets/vg/VG-SGG-dicts-with-attri.json'))
    idx2label = vocab_file['idx_to_label']
    label2idx = vocab_file['label_to_idx']
    idx2rel = vocab_file['idx_to_predicate']
    rel2idx = vocab_file['predicate_to_idx']
    print('-----------------------------id & lable & rel-----------------------------------------')

    detected_origin_path = './'
    detected_origin_result = torch.load(detected_origin_path + 'eval_results_relnew.pytorch')
    predictions = detected_origin_result['predictions']
    groundtruths = detected_origin_result['groundtruths']
    print('-----------------------------eval results---------------------------------------')
    detected_info = json.load(open(detected_origin_path + 'visual_info.json'))
    print('-----------------------------visual info----------------------------------------')  

    arti_caption = json.load(open('articaption.json'))
    print('-----------------------------captions----------------------------------------')

    newpres = []
    for i,prediction in enumerate(predictions):
        # print()
        print(i)
        captions = arti_caption[str(i)]
        # print(prediction.fields())
        prelabels = prediction.get_field('pred_labels').tolist()
        prescores = prediction.get_field('pred_scores').tolist()
        rel_pair = prediction.get_field('rel_pair_idxs').tolist()
        rel_scores = prediction.get_field('pred_rel_scores').tolist()
        rel_labels = prediction.get_field('pred_rel_labels').tolist()
        attribute_logits = prediction.get_field('attribute_logits')
        preattrs = []
        for iii in range(len(prelabels)):
            preattrs.append([])
        newpre = prediction.copy()
        reldict,objset = find(captions,idx2label,label2idx,rel2idx)
        # print(reldict)
        prelabels,prescores,rel_pair,rel_scores,rel_labels = add(prelabels,prescores,rel_pair,rel_scores,rel_labels,reldict,objset)

        newpre.add_field('pred_labels',torch.tensor(prelabels))
        newpre.add_field('pred_scores',torch.tensor(prescores))
        newpre.add_field('rel_pair_idxs',torch.tensor(rel_pair))
        newpre.add_field('pred_rel_scores',torch.tensor(rel_scores))
        newpre.add_field('pred_rel_labels',torch.tensor(rel_labels))
        newpre.add_field('attribute_logits',attribute_logits)

        newpres.append(newpre)

        # print(len(rel_pair))
        # if i == 3:break
    
    torch.save({'groundtruths':groundtruths, 'predictions':newpres}, detected_origin_path + 'eval_results_doublenew.pytorch')
    # torch.save({'groundtruths':groundtruths, 'predictions':newpres}, detected_origin_path + 'test.pytorch')

    return -1


def add(labels,scores,rel_pair,rel_scores,rel_labels,reldict,objset):
    # ch = 0
    for i,label in enumerate(labels):
        if label in objset:
            scores[i] = scores[i] + 0.5
            # ch = ch+1
    # print(ch)

    relcopy = {}
    for pair in reldict.keys():
        relcopy[pair] = set()
    
    for idx,pair in enumerate(rel_pair):
        subobj = labels[pair[0]]
        objobj = labels[pair[1]]
        if (subobj,objobj) in reldict.keys():
            if rel_labels[idx] in reldict[(subobj,objobj)]:
                rel_scores[idx][rel_labels[idx]] = rel_scores[idx][rel_labels[idx]] + 1
                relcopy[(subobj,objobj)].add(rel_labels[idx])
            else:
                rel_scores[idx][rel_labels[idx]] = rel_scores[idx][rel_labels[idx]] + 0.5
    
    for pair in reldict.keys():
        if len(reldict[pair])>len(relcopy[pair]):
            rel = list(reldict[pair]-relcopy[pair])
            sub = []
            for j,label in enumerate(labels):
                if label == pair[0]:
                    sub.append(j)
            obj = []
            for j,label in enumerate(labels):
                if label == pair[1]:
                    obj.append(j)
            for s in sub:
                for o in obj:
                    for r in rel:
                        rel_pair.append([s,o])
                        sco = [0]*51
                        sco[r] = 0.75
                        rel_scores.append(sco)
                        rel_labels.append(r)
    

    return labels,scores,rel_pair,rel_scores,rel_labels

def find(captions,idx2label,label2idx,rel2idx):
    lablist = list(idx2label.values())
    lablistvec = bc.encode(lablist,convert_to_tensor=True)

    rellist = []
    objset = set()
    for caption in captions:
        # print(len(objset))
        # print(caption)
        doc = nlp(caption)
        obj = []
        rel = []
        for token in doc:
            if token.ent_type_ == "label":
                obj.append((token.i,token.text.lower()))
            elif token.ent_type_ == "rel":
                rel.append((token.i,token.text.lower()))
        if len(obj)>=2:
            setnow = set([oi[0] for oi in obj])
            tripr = set()
            for r in rel:
                if doc[r[0]].dep_ == 'ROOT':
                    for child in doc[r[0]].children:
                        if (child.i in setnow)and child.dep_ == 'nsubj':
                            tripo1 = child.text.lower()
                        elif (child.i in setnow)and child.dep_ == 'dobj':
                            tripo2 = child.text.lower()
                    tripr.add(rel2idx[r[1]])
            tripo1 = label2idx[obj[0][1]]
            tripo2 = label2idx[obj[-1][1]]
            tripr = set()
            for r in rel:
                if r[0]>obj[0][0] and r[0]<obj[-1][0]:
                    tripr.add(rel2idx[r[1]])
            for o in obj[1:-1]:
                objset.add(label2idx[o[1]])
        else:
            for token in doc:
                if not(token._.is_entity) and (token.pos_ == 'NOUN'):
                    obj.append((token.i,token.text.lower()))
            if len(obj)>=2:
                noun = []
                for o in obj:
                    noun.append(o[1])
                nounvec = bc.encode(noun,convert_to_tensor=True)
                sim = util.pytorch_cos_sim(nounvec, lablistvec)
                nscore,nlab = sim.max(-1)
                nscore = nscore.tolist()
                nlab = [i+1 for i in nlab.tolist()]

                sorted_id = sorted(range(len(nscore)), key=lambda k: nscore[k], reverse=True)
                top1 = sorted_id[0]
                top2 = sorted_id[1]
                if obj[top1][0]>obj[top2][0]:
                    top1 = sorted_id[1]
                    top2 = sorted_id[0]
                tripo1 = nlab[top1]
                tripo2 = nlab[top2]
                tripr = set()
                for r in rel:
                    if r[0]>obj[top1][0] and r[0]<obj[top2][0]:
                        tripr.add(rel2idx[r[1]])
        if len(obj)>=2:
            objset.add(tripo1)
            objset.add(tripo2)
            rellist.append((tripo1,tripo2,tripr))
    
    reldict = {}
    for r in rellist:
        if (r[0],r[1]) in reldict.keys():
            reldict[(r[0],r[1])].union(r[2])
        else:
            reldict[(r[0],r[1])] = r[2]

    return reldict,objset

solve()
