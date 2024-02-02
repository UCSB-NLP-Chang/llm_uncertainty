import os
import re
import copy
import json
import numpy as np
np.set_printoptions(precision=3, suppress = True)
from src.common import gsm8k_extract_ans
from src.config import SAMPLE_N
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--log_path", type = str, required = True)
parser.add_argument("--output_path", type = str, required = True)
parser.add_argument("--answer_key", type = str, required = True)
parser.add_argument("--bnn", action='store_true')

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


filepath = args.log_path
with open(filepath, 'r', encoding='utf-8') as f:
    content = json.load(f)
best_n = SAMPLE_N

num_examples = len(content)

question_list = [x['question'] for x in content]
answer_list = [x['answer'] for x in content]
model_outputs_list = [x[args.answer_key] for x in content]


print("--------Uncertainty Quanficiation-----------")

is_multiple_ans = type(model_outputs_list[0][0]) == list
all_logs = []

for q_idx in range(num_examples):
    gt_ans = answer_list[q_idx]
    orig_q = question_list[q_idx]
    print("orig question:\n", orig_q)
    print(answer_list[q_idx])


    mv_answers = []
    cp_log_dict = copy.deepcopy(content[q_idx])

    if is_multiple_ans:
        curr_model_outputs_list = model_outputs_list[q_idx]
        curr_model_outputs_list = [[gsm8k_extract_ans(xx) for xx in x] for x in curr_model_outputs_list]
        ans2id = build_dict(curr_model_outputs_list)
        id2ans = {v:k for k,v in ans2id.items()}

        rewrite_freq_mat = []
        num_rewrite = len(curr_model_outputs_list)
        for rewrite_idx in range(num_rewrite):
            model_outputs = curr_model_outputs_list[rewrite_idx]
            rewrite_freq_array = np.zeros(len(ans2id))
            for idx, ans in enumerate(model_outputs):
                rewrite_freq_array[ans2id[ans]] += 1

            rewrite_freq_array = rewrite_freq_array / best_n
            rewrite_freq_mat.append(rewrite_freq_array)
        rewrite_freq_mat = np.stack(rewrite_freq_mat, axis = 0)

        print("GT: ", gt_ans)

        pred_posterior = np.mean(rewrite_freq_mat, axis = 0)
        posterior_entropy = compute_entropy(pred_posterior)

        if args.bnn:
            data_entropy_list = [compute_entropy(rewrite_freq_mat[i]) for i in range(rewrite_freq_mat.shape[0])]
            data_entropy_list = np.array(data_entropy_list)
            data_uncertainty =  np.mean(data_entropy_list)
            model_uncertainty = posterior_entropy - np.mean(knowledge_entropy_list)
            print("total uncertainty:", posterior_entropy)
            print("model uncertainty: ", posterior_entropy - np.mean(model_uncertainty))
            print()
        else:
            knowledge_entropy_list = [compute_entropy(rewrite_freq_mat[i]) for i in range(rewrite_freq_mat.shape[0])]
            knowledge_entropy_list = np.array(knowledge_entropy_list)
            print(knowledge_entropy_list)
            print("knowledge uncertainty", np.mean(knowledge_entropy_list))
            data_uncertainty = posterior_entropy - np.mean(knowledge_entropy_list)
            print("total uncertainty:", posterior_entropy)
            print("data uncertainty: ", posterior_entropy - np.mean(knowledge_entropy_list))
            print()

        most_freq_ans_id = np.argmax(pred_posterior)
        most_freq_ans = id2ans[most_freq_ans_id]

        cp_log_dict['most_freq_ans'] = most_freq_ans

        prop = data_uncertainty / (posterior_entropy + 1e-6)
        cp_log_dict['total_uncertainty'] = posterior_entropy
        cp_log_dict['data_uncertainty'] = data_uncertainty
        cp_log_dict['prop'] = prop
        

    else:
        model_outputs = model_outputs_list[q_idx]
        model_outputs = [gsm8k_extract_ans(x) for x in model_outputs]

        ans2id = build_dict([model_outputs])
        id2ans = {v:k for k,v in ans2id.items()}

        freq_array = np.zeros(len(ans2id))
        for idx, ans in enumerate(model_outputs):
            freq_array[ans2id[ans]] += 1

        freq_array = freq_array / best_n
        total_uncertainty = compute_entropy(freq_array)

        most_freq_ans_id = np.argmax(freq_array)
        most_freq_ans = id2ans[most_freq_ans_id]

        cp_log_dict['total_uncertainty'] = total_uncertainty
        cp_log_dict['most_freq_ans'] = most_freq_ans


    all_logs.append(cp_log_dict)

if not os.path.exists(os.path.dirname(args.output_path)): os.makedirs(os.path.dirname(args.output_path))

with open(args.output_path,'w',encoding='utf-8') as f:
    json.dump(all_logs, f, indent=4)

