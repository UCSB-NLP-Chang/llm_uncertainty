import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
import json
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score
from src.evaluation import recursive_normalize

filepath = 'logs/uq_eval/gpt_eval_nq.json'
with open(filepath, 'r', encoding='utf-8') as f:
    content = json.load(f)

model_ans_name = 'most_freq_ans'

pred_list = [x[model_ans_name] for x in content]
gt_list = [x['answer'] for x in content]

roc_list = []

print(len(pred_list))
refined_labels = []
uncertainty_list = []
corr_uncertainty = []
wrong_uncertainty = []
for i in range(len(pred_list)):
    mv_ans = content[i]['most_freq_ans']

    scores = [int(x) for x in content[i]['gptscore']]
    curr_score = max(scores)

    if curr_score > 70:
        refined_labels.append(True)
        corr_uncertainty.append(content[i]['total_uncertainty'])
    else:
        refined_labels.append(False)
        wrong_uncertainty.append(content[i]['total_uncertainty'])
    uncertainty_list.append(content[i]['total_uncertainty'])

refined_labels = np.array(refined_labels)
ys_array = refined_labels
xs_array = np.array(uncertainty_list)

auroc = roc_auc_score(refined_labels, -xs_array)
print("auroc:", auroc)
print()
roc_list.append(auroc)

print(roc_list)
print(np.mean(roc_list))

print("correct: ", np.mean(corr_uncertainty))
print("wrong: ", np.mean(wrong_uncertainty))

all_f1s = []
thres_cdts = np.arange(1,100) / 100
for thres in thres_cdts:
    pred_correctness_labels = np.array([True if x < thres else False for x in xs_array])
    tgt_correctness_labels = ys_array
    corr_f1 = f1_score(tgt_correctness_labels, pred_correctness_labels)
    all_f1s.append(corr_f1)
print("best f1: ", np.max(all_f1s))


