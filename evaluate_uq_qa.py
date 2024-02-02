import json
import os
from typing import Any
import numpy as np
np.set_printoptions(precision=3, suppress = True)
from src.common import majority_vote, remove_punctuation, check_answers
from src.evaluation import recursive_normalize
from src.config import SAMPLE_N
from jiwer import wer

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

def compute_acc(gts, preds):
    count = 0
    for i in range(len(gts)):
        if gts[i] == preds[i]:
            count += 1
    return count, count / len(preds)

class FuzzingDict():
    def __init__(self, ans2id: dict) -> None:
        self.ans2id = ans2id
        self.id2ans = {v:k for k,v in ans2id.items()}
        num_ans = len(self.ans2id)
        self.answer_list = [self.id2ans[x] for x in range(num_ans)]
    def __call__(self, key) -> Any:
        if key in self.ans2id:
            return self.ans2id[key]
        sim_answers = []
        
        for ans in self.answer_list:
            if ans in key or key in ans:
                sim_answers.append(ans)
        if len(sim_answers) == 1:
            final_ans = sim_answers[0]
            return self.ans2id[final_ans]
        elif len(sim_answers) == 0:
            final_ans = self.answer_list[0]
            return self.ans2id[final_ans]
        edit_distance = np.array([wer(key, ans) for ans in sim_answers])
        min_idx = np.argmin(edit_distance)
        selected = sim_answers[min_idx]
        return self.ans2id[selected]


filepath = args.log_path
with open(filepath, 'r', encoding='utf-8') as f:
    content = json.load(f)
best_n = SAMPLE_N

num_examples = len(content)

question_list = [x['question'] for x in content]
gt_answers = [x['answer'] for x in content]

print("--------Uncertainty Quanficiation-----------")

is_multiple_ans = type(content[0][args.answer_key][0]) == list
all_logs = []

for q_idx in range(num_examples):
    raw_output_label_sets = content[q_idx][args.answer_key]
    raw_output_label_sets = [check_answers(x) for x in raw_output_label_sets]
    raw_output_label_sets = recursive_normalize(raw_output_label_sets)

    unique_output_labels = []
    for raw_output_label_set in raw_output_label_sets:
        output_labels = sorted(raw_output_label_set, key=lambda x: len(x), reverse=True)
        for x in output_labels:
            exist_flag = False
            for exist_x in unique_output_labels:
                if x in exist_x:
                    exist_flag = True
                    break
            if not exist_flag:
                unique_output_labels.append(x)


    output_labels = unique_output_labels

    ans2idx = {ans: ans_id for ans_id,ans in enumerate(output_labels)}
    idx2ans = {v:k for k,v in ans2idx.items()}

    gt_ans = gt_answers[q_idx]
    gt_ans = recursive_normalize(gt_ans)
    orig_q = question_list[q_idx]

    num_rewrite = len(raw_output_label_sets)
    if num_rewrite == 0:
        raise NotImplementedError

    all_cdt_answers = []
    for rewrite_idx in range(num_rewrite):
        rewrite_answer_lists = raw_output_label_sets[rewrite_idx]
        all_cdt_answers = all_cdt_answers + rewrite_answer_lists

    rewrite_freq_mat = []
    mv_answers = []

    all_unk = False

    fuzz_dict = FuzzingDict(ans2idx)
    print(ans2idx)

    output_space_size = (len(idx2ans))
    for rewrite_idx in range(num_rewrite):
        rewrite_answer_list = raw_output_label_sets[rewrite_idx]
        print(rewrite_answer_list)
        if rewrite_answer_list is None:
            all_unk = True
            break

        rewrite_freq_array = np.zeros(len(idx2ans))
        if len(idx2ans) <= 1 and 'unknown' in rewrite_answer_list: 
            ### special case 1:
            ### there is only one answer in the output given the current clarification
            ### and the answer is "unknown", means this clarification is very possible to be invalid
            ### for example, the model clarify "when did lionel messi when the world cup" as "when did lionel messi when the world cup in 2015"
            ### So we skip this clarification
            continue

        for idx, ans in enumerate(rewrite_answer_list):
            if ans == "unknown":
                ### special case 2:
                ### there are multiple different answers in the output given the current clarification
                ### and the current answer is "unknown"
                ### in this case, we regard it as epistemic uncertainty (i.e., the model directly answer the question with "sorry i do not know" or similar answers)
                ### to handle this case, we add the frequency of other answers (except unknown) by 1 / (total different answers - 1) and leave the frequency of "unknown" unchanged
                rewrite_freq_array = rewrite_freq_array + 1/(output_space_size - 1)
                rewrite_freq_array[ans2idx['unknown']] = rewrite_freq_array[ans2idx['unknown']] - 1/(output_space_size - 1)
            else:
                rewrite_freq_array[fuzz_dict(ans)] += 1
                
        rewrite_freq_array = rewrite_freq_array / best_n
        rewrite_freq_mat.append(rewrite_freq_array)
        mv_ans = majority_vote(rewrite_answer_list)[0]
        mv_answers.append(mv_ans)


    if all_unk or len(rewrite_freq_mat) == 0:
        ### if all clarifications belongs to special case 1
        ### that means the clarificaiton LLM gives us a series of invalid clarifications
        ### in such a case, we regard the LLM does not have enough knowledge for this question
        ### and we manually set the epistemic (model) uncertainty as 1 and aleatoric (data) uncertainty as 0
        posterior_entropy = 1
        data_uncertainty = 0
        model_uncertainty = posterior_entropy - data_uncertainty

        if args.bnn:
            data_uncertainty, model_uncertainty = model_uncertainty, data_uncertainty
        log_dict = {
            'question': orig_q, 
            'answer': gt_ans,
            'ext_all_ans': content[q_idx][args.answer_key],
            'data_uncertainty':data_uncertainty, 
            "total_uncertainty": posterior_entropy,
            'model_uncertainty': model_uncertainty,
            'model_uncertainty_list': [1 for _ in range(len(content[q_idx][args.answer_key]))],
            "label": content[q_idx]['label'] if 'label' in content[q_idx] else None,
            'most_freq_ans': "unknown",
            }
        all_logs.append(log_dict)
        continue
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

    most_freq_id = np.argmax(pred_posterior)
    most_freq_ans = fuzz_dict.id2ans[most_freq_id]

    data_uncertainty = posterior_entropy - np.mean(knowledge_entropy_list)

    print("total uncertainty:", posterior_entropy)
    print("data uncertainty: ", posterior_entropy - np.mean(knowledge_entropy_list))
    print()

    model_uncertainty = posterior_entropy - data_uncertainty
    if args.bnn:
        data_uncertainty, model_uncertainty = model_uncertainty, data_uncertainty
    log_dict = {
        'question': orig_q, 
        'answer': gt_ans,
        'ext_all_ans': content[q_idx][args.answer_key],
        'data_uncertainty':data_uncertainty, 
        "total_uncertainty": posterior_entropy,
        'model_uncertainty': model_uncertainty,
        'model_uncertainty_list': knowledge_entropy_list.tolist(),
        "label": content[q_idx]['label'] if 'label' in content[q_idx] else None,
        'most_freq_ans': most_freq_ans,
        }
    all_logs.append(log_dict)

if not os.path.exists(os.path.dirname(args.output_path)):
    os.makedirs(os.path.dirname(args.output_path))

with open(args.output_path,'w',encoding='utf-8') as f:
    json.dump(all_logs, f, indent=4)


