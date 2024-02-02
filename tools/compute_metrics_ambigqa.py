import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
import json
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, roc_curve
from src.evaluation import is_ambig

filepath = 'logs/uq_eval/ambigqa.json'
with open(filepath, 'r', encoding='utf-8') as f:
    content = json.load(f)

print(len(content))

roc_list = []

refined_labels = []
ambig_uncertainty = []
unambig_uncertainty = []
uncertainty_list = []
for i in range(len(content)):
    labels = content[i]['label']
    ambig_flag = is_ambig(labels)
    if ambig_flag:
        refined_labels.append(True)
        ambig_uncertainty.append(content[i]['data_uncertainty'])
    else:
        refined_labels.append(False)
        unambig_uncertainty.append(content[i]['data_uncertainty'])
    uncertainty_list.append(content[i]['data_uncertainty'])


refined_labels = np.array(refined_labels)
ys_array = refined_labels
xs_array = np.array(uncertainty_list)

auroc = roc_auc_score(refined_labels, xs_array)
print("auroc:", auroc)
print()
roc_list.append(auroc)

print(roc_list)
print(np.mean(roc_list))

print("ambig: ", np.mean(ambig_uncertainty))
print("unambig: ", np.mean(unambig_uncertainty))

all_f1s = []
all_precisions = []
all_recalls = []
thres_cdts = np.arange(1,100) / 100
for thres in thres_cdts:
    pred_correctness_labels = np.array([True if x > thres else False for x in xs_array])
    tgt_correctness_labels = ys_array
    corr_f1 = f1_score(tgt_correctness_labels, pred_correctness_labels)
    precision = precision_score(tgt_correctness_labels, pred_correctness_labels)
    recall = recall_score(tgt_correctness_labels, pred_correctness_labels)
    
    all_precisions.append(precision)
    all_recalls.append(recall)

    all_f1s.append(corr_f1)

best_idx = np.argmax(all_f1s)
print("best f1: ", np.max(all_f1s))
print('best precision: ', all_precisions[best_idx])
print("best recall: ", all_recalls[best_idx])
print("best thres: ", thres_cdts[best_idx])

best_thres = thres_cdts[best_idx]
ambig_preds = np.array([x > best_thres for x in ambig_uncertainty])
unambig_preds = np.array([x <= best_thres for x in unambig_uncertainty])


ambig_pred_acc = np.sum(ambig_preds) / len(ambig_uncertainty)
print("ambig acc: ", ambig_pred_acc)

unambig_pred_acc = np.sum(unambig_preds) / len(unambig_uncertainty)
print("unambig acc: ", unambig_pred_acc)


