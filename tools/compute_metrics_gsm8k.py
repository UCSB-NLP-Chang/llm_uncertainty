import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
import json
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score
from src.common import gsm8k_extract_ans

filepath = 'logs/uq_eval/gsm8k.json'
with open(filepath, 'r', encoding='utf-8') as f:
    content = json.load(f)

gt_list = [x['answer'] for x in content]

roc_list = []

refined_labels = []
corr_uncertainty = []
wrong_uncertainty = []
for i in range(len(gt_list)):
    gt_ans = gsm8k_extract_ans(gt_list[i])
    curr_answer = content[i]['most_freq_ans']
    if curr_answer == gt_ans:
        refined_labels.append(True)
        corr_uncertainty.append(content[i]['total_uncertainty'])
    else:
        refined_labels.append(False)
        wrong_uncertainty.append(content[i]['total_uncertainty'])

refined_labels = np.array(refined_labels)
ys_array = refined_labels
print('acc: ', np.mean(ys_array))

tu_array = np.array([x['total_uncertainty'] for x in content])
xs_array = tu_array

auroc = roc_auc_score(refined_labels, -xs_array)
print("auroc:", auroc)
print()
print("correct: ", np.mean(corr_uncertainty))
print("wrong: ", np.mean(wrong_uncertainty))

all_f1s = []
thres_cdts = np.arange(1,200) / 100
for thres in thres_cdts:
    pred_mistake_labels = np.array([True if x >= thres else False for x in xs_array])
    tgt_mistake_labels = np.array([True if not x else False for x in ys_array])
    corr_f1 = f1_score(tgt_mistake_labels, pred_mistake_labels)

    all_f1s.append(corr_f1)
print("best f1: ", np.max(all_f1s))


