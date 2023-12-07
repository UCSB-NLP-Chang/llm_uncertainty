import json
import os
from typing import Any
import numpy as np
import string
np.set_printoptions(precision=3, suppress = True)
from jiwer import wer
from src.evaluation import recursive_normalize
from src.common import ambiginst_extract_ans

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--log_path", type = str, required = True)
parser.add_argument("--output_path", type = str, required = True)
parser.add_argument("--answer_key", type = str, required = True)
args = parser.parse_args()


def build_dict(list_of_list):
    item2id = {}
    for curr_list in list_of_list:
        for item in curr_list:
            if item not in item2id:
                item2id[item] = len(item2id)
    return item2id

def compute_entropy(vec: np.ndarray):
    vec = vec + 1e-10
    vec = vec / np.sum(vec)
    entropy = -np.sum(vec * np.log2(vec))
    return entropy

def compute_acc(gts, preds):
    count = 0
    for i in range(len(gts)):
        if gts[i] == preds[i]:
            count += 1
            # print(gts[i], preds[i])
    return count, count / len(preds)

def majority_vote(answers):
    ans2freq = {}
    max_freq = 0
    max_ans = None
    for ans in answers:
        if ans not in ans2freq:
            ans2freq[ans] = 1
        else:
            ans2freq[ans] += 1
        if ans2freq[ans] > max_freq:
            max_ans = ans
            max_freq = ans2freq[ans]
    return max_ans, max_freq

def process_ans(ans:str):
    return ans.strip(string.punctuation)


with open(args.log_path, 'r', encoding='utf-8') as f:
    content = json.load(f)


best_n = 10

num_examples = len(content)


print("--------Uncertainty Quanficiation-----------")

all_logs = []

for q_idx in range(num_examples):
    curr_log_dict = content[q_idx]
    task_desc = content[q_idx]['orig_instruction']
    if 'isambig' in content[q_idx]:
        inst_ambig = content[q_idx]['isambig']
    else:
        inst_ambig = True

    raw_output_label_sets = curr_log_dict[args.answer_key]
    raw_output_label_sets = [[ambiginst_extract_ans(x) for x in xx] for xx in raw_output_label_sets]

    raw_output_label_sets = recursive_normalize(raw_output_label_sets)

    ans2idx = build_dict(raw_output_label_sets)
    idx2ans = {v:k for k,v in ans2idx.items()}

    gt_ans = content[q_idx]['target']
    orig_q = content[q_idx]['input']

    print("Task: ", task_desc)
    print("orig question:\n", orig_q)

    curr_all_rewrite_cots = raw_output_label_sets
    num_rewrite = len(curr_all_rewrite_cots)
    if num_rewrite == 0:
        posterior_entropy = 1
        data_uncertainty = 0
        prop = 0
        log_dict = {
            'question': orig_q, 
            'answer': gt_ans,
            'rewrite_all_ans': raw_output_label_sets,
            'prop': prop, 
            'data_uncertainty':data_uncertainty, 
            "total_uncertainty": posterior_entropy,
            'model_uncertainty_list': [1 for _ in range(len(curr_log_dict))],
            "isambig": inst_ambig,
            }
        all_logs.append(log_dict)
        continue

    mv_answers = []
    output_space_size = (len(idx2ans))
    rewrite_freq_mat = []
    for rewrite_idx in range(num_rewrite):
        rewrite_answer_list = curr_all_rewrite_cots[rewrite_idx]

        rewrite_freq_array = np.zeros(len(idx2ans))
        for idx, ans in enumerate(rewrite_answer_list):
            rewrite_freq_array[ans2idx[ans]] += 1

        rewrite_freq_array = rewrite_freq_array / best_n
        rewrite_freq_mat.append(rewrite_freq_array)
        mv_ans = majority_vote(rewrite_answer_list)[0]
        mv_answers.append(mv_ans)


    rewrite_freq_mat = np.stack(rewrite_freq_mat, axis = 0)


    knowledge_entropy_list = [compute_entropy(rewrite_freq_mat[i]) for i in range(rewrite_freq_mat.shape[0])]
    print("num set: ", len(idx2ans))
    print("GT: ", gt_ans)
    print("MV: ", mv_answers)
    knowledge_entropy_list = np.array(knowledge_entropy_list)
    print(knowledge_entropy_list)
    print("knowledge uncertainty", np.mean(knowledge_entropy_list))

    pred_posterior = np.mean(rewrite_freq_mat, axis = 0)
    posterior_entropy = compute_entropy(pred_posterior)


    data_uncertainty = posterior_entropy - np.mean(knowledge_entropy_list)

    print("total uncertainty:", posterior_entropy)
    print("data uncertainty: ", posterior_entropy - np.mean(knowledge_entropy_list))
    print()

    prop = data_uncertainty / (posterior_entropy + 1e-6)
    log_dict = {
        'question': orig_q, 
        'answer': gt_ans,
        'rewrite_all_ans': raw_output_label_sets,
        'prop': prop, 
        'data_uncertainty':data_uncertainty, 
        "total_uncertainty": posterior_entropy,
        'model_uncertainty_list': knowledge_entropy_list.tolist(),
        'isambig': inst_ambig
        }
    all_logs.append(log_dict)


if not os.path.exists(os.path.dirname(args.output_path)):
    os.makedirs(os.path.dirname(args.output_path))

with open(args.output_path,'w',encoding='utf-8') as f:
    json.dump(all_logs, f, indent=4)

du_list = [x['data_uncertainty'] for x in all_logs]
print("average data uncertainty: ", np.mean(du_list))

