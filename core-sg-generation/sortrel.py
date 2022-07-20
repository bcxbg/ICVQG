import torch

print("relnew")
detected_origin_path = './'
detected_origin_result = torch.load(detected_origin_path + 'eval_results_doublenew.pytorch')
predictions = detected_origin_result['predictions']
groundtruths = detected_origin_result['groundtruths']
print('ok')

newpres = []
for i,prediction in enumerate(predictions):
    # print()
    print(i)
    prelabels = prediction.get_field('pred_labels')
    prescores = prediction.get_field('pred_scores').tolist()
    rel_pair = prediction.get_field('rel_pair_idxs').tolist()
    rel_scores = prediction.get_field('pred_rel_scores').tolist()
    rel_labels = prediction.get_field('pred_rel_labels').tolist()
    attribute_logits = prediction.get_field('attribute_logits')

    pred_scores = [max(k[1:]) for k in rel_scores]

    sorted_id = sorted(range(len(pred_scores)), key=lambda k: pred_scores[k]*prescores[rel_pair[k][0]]*prescores[rel_pair[k][1]], reverse=True)
    
    rel_pair = [rel_pair[k] for k in sorted_id]
    rel_scores = [rel_scores[k] for k in sorted_id]
    rel_labels = [rel_labels[k] for k in sorted_id]
    
    newpre = prediction.copy()       
    newpre.add_field('pred_labels',prelabels)
    newpre.add_field('pred_scores',torch.tensor(prescores))
    newpre.add_field('rel_pair_idxs',torch.tensor(rel_pair))
    newpre.add_field('pred_rel_scores',torch.tensor(rel_scores))
    newpre.add_field('pred_rel_labels',torch.tensor(rel_labels))
    newpre.add_field('attribute_logits',attribute_logits)

    newpres.append(newpre)

    # print(sorted_id[0:30])
    # if i == 2:break
    
torch.save({'groundtruths':groundtruths, 'predictions':newpres}, detected_origin_path + 'eval_results_doublesorted.pytorch')
# torch.save({'groundtruths':groundtruths, 'predictions':newpres}, detected_origin_path + 'test.pytorch')
print('down')

