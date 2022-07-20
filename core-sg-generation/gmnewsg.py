import torch
import json
import h5py
import numpy as np
import copy
from matplotlib.pyplot import imshow
from PIL import Image, ImageDraw
from IPython.display import display

import torch.nn.functional as F
import pdb, os, argparse
from scipy import misc
import imageio
from model.CPD_models import CPD_VGG
from model.CPD_ResNet_models import CPD_ResNet
from data1 import test_dataset

# badimgbox = {4252}
badimgbox = {}
print('go')

vocab_file = json.load(open('./datasets/vg/VG-SGG-dicts-with-attri.json'))

print('ok1')
# remove invalid image
# corrupted_ims = [1592, 1722, 4616, 4617]
# tmp = []
# for item in image_file:
#     if int(item['image_id']) not in corrupted_ims:
#         tmp.append(item)
# image_file = tmp
# print(len(image_file),' images')

# load detected results
detected_origin_path = './'

print('1')
detected_origin_result = torch.load(detected_origin_path + 'eval_results_relnew.pytorch')
print('2')
detected_info = json.load(open(detected_origin_path + 'visual_info.json'))
print('ok2')

# load cpd
model = CPD_VGG()
model.load_state_dict(torch.load('CPD.pth'))
model.cuda()
model.eval()
save_path = detected_origin_path + 'cpd_results/4/'
if not os.path.exists(save_path):
  os.makedirs(save_path)

#attr
attlist = {
  "color":{1,2,3,4,5,6,7,11,12,13,16,17,20,25,34,39,42,44,47,102,107,108,111,149,160,180,181,197},
  "pattern":{31,45,51,78,83,87,100,109,123,141,156,163,184,185},
  "size":{8,9,14,15,26,52,57,58,59,112,138,143,152,194},
  "material":{10,21,23,27,32,40,50,63,67,69,70,72,81,88,94,104,114,122,128,137,142,161,167,171,173,178,188},
  "state":{18,24,28,29,30,33,35,36,37,38,41,46,49,53,54,55,56,60,61,62,64,66,68,71,73,74,75,76,79,80,82,84,85,86,89,90,91,92,95,96,97,99,
           101,103,105,106,110,113,115,117,118,119,120,124,125,127,129,131,132,133,134,135,136,139,140,144,145,146,147,148,150,
           154,155,157,158,159,162,164,165,166,168,169,170,172,174,176,177,183,186,187,189,191,192,193,195,196,198,199,200},
  "shape":{19,48,65,77,116,121,126,175,179},
  "det":{43,98,151,153,182,190},
  "backdet":{22,93,130}
}



def get_info_by_idx(idx, det_input, thres=0.1):
    #print('getscene')
    scene = {}
    scene["image_index"] = idx
    objs = []
    relas = []
    # groundtruth = det_input['groundtruths'][idx]
    prediction = det_input['predictions'][idx]
    #print(groundtruth.fields())
    #print(prediction.fields())
    # image path
    img_path = detected_info[idx]['img_file']
    scene["image_filename"] = img_path
    # print(img_path)
    #cpd
    cpd_loader = test_dataset(img_path, 352)
    image, name ,ysize = cpd_loader.load_data()
    nsize = (ysize[1],ysize[0])
    # print(prediction.size)
    # print(name,nsize)
    image = image.cuda()
    _, resim = model(image)
    resim = F.interpolate(resim, size=nsize, mode='bilinear', align_corners=False)
    resim = resim.sigmoid().data.cpu().numpy().squeeze()
    resim = (resim - resim.min()) / (resim.max() - resim.min() + 1e-8) 
    imageio.imsave(save_path+name, resim)
    res = resim.T
    #print(res.max(),res.min())
    # print(res.shape)
    #misc.imsave(save_path+name, res)

    # boxes
    # boxes = groundtruth.bbox
    boxes = prediction.bbox
    #print(boxes)
    # object labels
    idx2label = vocab_file['idx_to_label']
    # labels = [idx2label[str(i)] for i in groundtruth.get_field('labels').tolist()]
    #print('labels:  ',labels)
    pred_labels = [idx2label[str(i)] for i in prediction.get_field('pred_labels').tolist()]
    #print('pred_labels:  ',pred_labels)
    pred_scores = prediction.get_field('pred_scores').tolist()
    #print('pred_scores:  ',pred_scores)
    idx2attr = vocab_file['idx_to_attribute']
    attrslgt = prediction.get_field('attribute_logits').tolist()
    attrslgt[:][0] = 0
    attrs = []
    for a in attrslgt:
      sorted_id = sorted(range(len(a)), key=lambda k: a[k], reverse=True)
      item = []
      for sidx in reversed(sorted_id[0:10]):
        if a[sidx] >= 0.1:
          item.append(sidx)
      attrs.append(item)
      


    #print('attrs:  ',attrs)

    light = []
    num = 0
    for box,label,score,attr in zip(boxes,pred_labels,pred_scores,attrs):
      x1,y1,x2,y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
      cpdsum = 0
      for i in range(x1,x2):
        for j in range(y1,y2):
          cpdsum = cpdsum + res[i,j]
      if ((x2-x1)*(y2-y1) > 0)and(cpdsum/((x2-x1)*(y2-y1)) > 0.8):
        light.append(num)
        num = num + 1
        obj = {}
        obj["label"] = label
        obj["box"] = [x1,y1,x2,y2]
        for key in attlist.keys():
          if key == "det":
            obj[key] = "the"
          else:
            obj[key] = None
        # obj["score"] = score
        for ati in attr:
          for key in attlist.keys():
            if ati in attlist[key]:
              obj[key] = idx2attr[str(ati)]
              break
        objs.append(obj)
      else:
        light.append(-1)
    #print(objs,light)
    if num == 0:
      print("!!!!!warning!!!!!",idx)
      light = []
      num = 0
      for box,label,score,attr in zip(boxes,pred_labels,pred_scores,attrs):
        x1,y1,x2,y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        cpdsum = 0
        for i in range(x1,x2):
          for j in range(y1,y2):
            cpdsum = cpdsum + res[i,j]
        if ((x2-x1)*(y2-y1) > 0)and(cpdsum/((x2-x1)*(y2-y1)) > 0.5):
          light.append(num)
          num = num + 1
          obj = {}
          obj["label"] = label
          obj["box"] = [x1,y1,x2,y2]
          for key in attlist.keys():
            if key == "det":
              obj[key] = "the"
            else:
              obj[key] = None
          # obj["score"] = score
          for ati in attr:
            for key in attlist.keys():
              if ati in attlist[key]:
                obj[key] = idx2attr[str(ati)]
                break
          objs.append(obj)
        else:
          light.append(-1)
    if num == 0:
      print("!!!!!no cpd!!!!!",idx)


    # groundtruth relation triplet
    idx2pred = vocab_file['idx_to_predicate']
    # gt_rels = groundtruth.get_field('relation_tuple').tolist()
    # gt_rels = [(labels[i[0]], idx2pred[str(i[2])], labels[i[1]]) for i in gt_rels]
    #print('gt_rels:  ',gt_rels)
    # prediction relation triplet
    pred_rel_pair = prediction.get_field('rel_pair_idxs').tolist()
    #print('pred_rel_pair:  ',pred_rel_pair)
    pred_rel_label = prediction.get_field('pred_rel_scores')
    pred_rel_label[:,0] = 0
    pred_rel_score, pred_rel_label = pred_rel_label.max(-1)
    pred_rels = [(pred_labels[i[0]], idx2pred[str(j)], pred_labels[i[1]]) for i, j in zip(pred_rel_pair, pred_rel_label.tolist())]
    
    #scene1cpd范围内结点关系
    lightrel = []
    numrel = 0
    cpdnum = num
    for pair,rel,score in zip(pred_rel_pair,pred_rels,pred_rel_score):
      if score>=thres and ((light[pair[0]]>=0 and light[pair[0]]<cpdnum) and (light[pair[1]]>=0 and light[pair[1]]<cpdnum)):
        lightrel.append(numrel)
        numrel = numrel + 1
        rela = {}
        rela["predicate"] = rel[1]
        rela["subject_idx"] = light[pair[0]]
        rela["object_idx"] = light[pair[1]]
        relas.append(rela)
      else:
        lightrel.append(-1)
    scene1 = copy.deepcopy(scene)
    scene1["objects"] = objs[:]
    scene1["relationships"] = relas[:]
    #scene2cpd范围+一跳结点和关系
    cpdnum = num
    i = -1
    for pair,rel,score in zip(pred_rel_pair,pred_rels,pred_rel_score):
      i = i + 1
      if lightrel[i]==-1 and score>=thres:
        if ((light[pair[0]]>=0 and light[pair[0]]<cpdnum) or (light[pair[1]]>=0 and light[pair[1]]<cpdnum)):
          if light[pair[0]]==-1:
            obj = {}
            obj["label"] = pred_labels[pair[0]]
            x1,y1,x2,y2 = int(boxes[pair[0]][0]), int(boxes[pair[0]][1]), int(boxes[pair[0]][2]), int(boxes[pair[0]][3])
            obj["box"] = [x1,y1,x2,y2]
            attr = attrs[pair[0]]
            for key in attlist.keys():
              if key == "det":
                obj[key] = "the"
              else:
                obj[key] = None
            for ati in attr:
              for key in attlist.keys():
                if ati in attlist[key]:
                  obj[key] = idx2attr[str(ati)]
                  break
            # obj["score"] = pred_scores[pair[0]]
            objs.append(obj)
            light[pair[0]] = num
            num = num + 1
          if light[pair[1]]==-1:
            obj = {}
            obj["label"] = pred_labels[pair[1]]
            x1,y1,x2,y2 = int(boxes[pair[1]][0]), int(boxes[pair[1]][1]), int(boxes[pair[1]][2]), int(boxes[pair[1]][3])
            obj["box"] = [x1,y1,x2,y2]
            # obj["score"] = pred_scores[pair[1]]
            attr = attrs[pair[1]]
            for key in attlist.keys():
              if key == "det":
                obj[key] = "the"
              else:
                obj[key] = None
            for ati in attr:
              for key in attlist.keys():
                if ati in attlist[key]:
                  obj[key] = idx2attr[str(ati)]
                  break
            objs.append(obj)
            light[pair[1]] = num
            num = num + 1
          rela = {}
          rela["predicate"] = rel[1]
          rela["subject_idx"] = light[pair[0]]
          rela["object_idx"] = light[pair[1]]
          relas.append(rela)
          lightrel.append(numrel)
          numrel = numrel + 1
    scene2 = copy.deepcopy(scene)
    scene2["objects"] = objs[:]
    scene2["relationships"] = relas[:]
    #scene3一跳范围内结点关系
    cpdnum = num
    i = -1
    for pair,rel,score in zip(pred_rel_pair,pred_rels,pred_rel_score):
      i = i + 1
      if lightrel[i]==-1 and score>=thres:
        if ((light[pair[0]]>=0 and light[pair[0]]<cpdnum) and (light[pair[1]]>=0 and light[pair[1]]<cpdnum)):
          rela = {}
          rela["predicate"] = rel[1]
          rela["subject_idx"] = light[pair[0]]
          rela["object_idx"] = light[pair[1]]
          relas.append(rela)
          lightrel.append(numrel)
          numrel = numrel + 1
    scene3 = copy.deepcopy(scene)
    scene3["objects"] = objs[:]
    scene3["relationships"] = relas[:]
    #scene4一跳范围+二跳结点和关系
    cpdnum = num
    i = -1
    for pair,rel,score in zip(pred_rel_pair,pred_rels,pred_rel_score):
      i = i + 1
      if lightrel[i]==-1 and score>=thres:
        if ((light[pair[0]]>=0 and light[pair[0]]<cpdnum) or (light[pair[1]]>=0 and light[pair[1]]<cpdnum)):
          if light[pair[0]]==-1:
            obj = {}
            obj["label"] = pred_labels[pair[0]]
            x1,y1,x2,y2 = int(boxes[pair[0]][0]), int(boxes[pair[0]][1]), int(boxes[pair[0]][2]), int(boxes[pair[0]][3])
            obj["box"] = [x1,y1,x2,y2]
            # obj["score"] = pred_scores[pair[0]]
            attr = attrs[pair[0]]
            for key in attlist.keys():
              if key == "det":
                obj[key] = "the"
              else:
                obj[key] = None
            for ati in attr:
              for key in attlist.keys():
                if ati in attlist[key]:
                  obj[key] = idx2attr[str(ati)]
                  break
            objs.append(obj)
            light[pair[0]] = num
            num = num + 1
          if light[pair[1]]==-1:
            obj = {}
            obj["label"] = pred_labels[pair[1]]
            x1,y1,x2,y2 = int(boxes[pair[1]][0]), int(boxes[pair[1]][1]), int(boxes[pair[1]][2]), int(boxes[pair[1]][3])
            obj["box"] = [x1,y1,x2,y2]
            # obj["score"] = pred_scores[pair[1]]
            attr = attrs[pair[1]]
            for key in attlist.keys():
              if key == "det":
                obj[key] = "the"
              else:
                obj[key] = None
            for ati in attr:
              for key in attlist.keys():
                if ati in attlist[key]:
                  obj[key] = idx2attr[str(ati)]
                  break
            objs.append(obj)
            light[pair[1]] = num
            num = num + 1
          rela = {}
          rela["predicate"] = rel[1]
          rela["subject_idx"] = light[pair[0]]
          rela["object_idx"] = light[pair[1]]
          relas.append(rela)
          lightrel.append(numrel)
          numrel = numrel + 1
    scene4 = copy.deepcopy(scene)
    scene4["objects"] = objs[:]
    scene4["relationships"] = relas[:]
    #scene5二跳范围内结点关系
    cpdnum = num
    i = -1
    for pair,rel,score in zip(pred_rel_pair,pred_rels,pred_rel_score):
      i = i + 1
      if lightrel[i]==-1 and score>=thres:
        if ((light[pair[0]]>=0 and light[pair[0]]<cpdnum) and (light[pair[1]]>=0 and light[pair[1]]<cpdnum)):
          rela = {}
          rela["predicate"] = rel[1]
          rela["subject_idx"] = light[pair[0]]
          rela["object_idx"] = light[pair[1]]
          relas.append(rela)
          lightrel.append(numrel)
          numrel = numrel + 1
    scene5 = copy.deepcopy(scene)
    scene5["objects"] = objs[:]
    scene5["relationships"] = relas[:]
    return scene1,scene2,scene3,scene4,scene5






def show_all(start_idx, length):
    info={
      "split": "train",
      "version": "1.0",
      "date": "11/8/2021"
    }
    scenes1=[]
    scenes2=[]
    scenes3=[]
    scenes4=[]
    scenes5=[]
    for cand_idx in range(start_idx, start_idx+length):
        print(cand_idx)
        if not(cand_idx in badimgbox):
          #draw_image(*get_info_by_idx(cand_idx, detected_origin_result))
          scene1,scene2,scene3,scene4,scene5 = get_info_by_idx(cand_idx, detected_origin_result)
          scenes1.append(scene1)
          scenes2.append(scene2)
          scenes3.append(scene3)
          scenes4.append(scene4)
          scenes5.append(scene5)
          
    test = {}
    test["info"] = info
    test1 = copy.deepcopy(test)
    test1["scenes"] = scenes1
    test2 = copy.deepcopy(test)
    test2["scenes"] = scenes2
    test3 = copy.deepcopy(test)
    test3["scenes"] = scenes3
    test4 = copy.deepcopy(test)
    test4["scenes"] = scenes4
    test5 = copy.deepcopy(test)
    test5["scenes"] = scenes5
    return test1,test2,test3,test4,test5

print("start")
totalnum = len(detected_origin_result['predictions'])
# test1,test2,test3,test4,test5 = show_all(start_idx=0, length=13000)
# test1,test2,test3,test4,test5 = show_all(start_idx=13000, length=totalnum-13000)
# test1,test2,test3,test4,test5 = show_all(start_idx=0, length=14500)
# test1,test2,test3,test4,test5 = show_all(start_idx=14500, length=14500)
# test1,test2,test3,test4,test5 = show_all(start_idx=29000, length=14500)
test1,test2,test3,test4,test5 = show_all(start_idx=43500, length=totalnum-43500)
# test1,test2,test3,test4,test5 = show_all(start_idx=0, length=totalnum)
# test1,test2,test3,test4,test5 = show_all(start_idx=0, length=3)
# t=2
# print(len(test1["scenes"][t]["objects"]))
# print(len(test1["scenes"][t]["relationships"]))
# print(len(test2["scenes"][t]["objects"]))
# print(len(test2["scenes"][t]["relationships"]))
# print(len(test3["scenes"][t]["objects"]))
# print(len(test3["scenes"][t]["relationships"]))
# print(len(test4["scenes"][t]["objects"]))
# print(len(test4["scenes"][t]["relationships"]))
# print(len(test5["scenes"][t]["objects"]))
# print(len(test5["scenes"][t]["relationships"]))
#print(sceneresult)

scenetest_save_path = detected_origin_path + 'scene_results_all/4/'
with open(scenetest_save_path + "scene1.json","w") as f:
  json.dump(test1,f)
with open(scenetest_save_path + "scene2.json","w") as f:
  json.dump(test2,f)
with open(scenetest_save_path + "scene3.json","w") as f:
  json.dump(test3,f)
with open(scenetest_save_path + "scene4.json","w") as f:
  json.dump(test4,f)
with open(scenetest_save_path + "scene5.json","w") as f:
  json.dump(test5,f)

print("down")