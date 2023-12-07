import re
import os
import random
from tqdm.auto import tqdm as auto_tqdm
import copy
import json
import numpy as np
import argparse
from src.data_util import load_data
from src.prompt_util import load_fewshot_prompt, load_system_prompt, inst_transform
from src.config import MAX_CLARIFICATION, SAMPLE_N
from src.common import completion_with_backoff

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", type = str, required = True)
parser.add_argument("--clarification_path", type = str, required = True)
parser.add_argument("--output_path", type = str, required = True)
args = parser.parse_args()


def format_message(system_prompt, user_prompt):
    if len(system_prompt) > 1:
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    else:
        messages=[
            {"role": "user", "content": user_prompt},
        ]
    return messages

def format_query(log_dict, dataset_name, prompt):
    if dataset_name == 'ambigqa' or dataset_name == 'nq_open':
        orig_q = log_dict['question']
        rewrite_qs = log_dict['self_clarification']
        if len(rewrite_qs) == 0:
            rewrite_qs = [orig_q]
        unique_rewrite_qs = []
        for item in rewrite_qs:
            if item not in unique_rewrite_qs:
                unique_rewrite_qs.append(item)
        print(len(rewrite_qs), "->", len(unique_rewrite_qs))
        rewrite_qs = unique_rewrite_qs[:MAX_CLARIFICATION]


        clarified_inputs = [prompt.strip() + '\n\nQ: ' + x for x in rewrite_qs]
    elif dataset_name == 'ambig_inst':
        orig_inst = log_dict['orig_instruction']
        all_clarifications = log_dict['self_clarification']
        q = log_dict['input']
        if len(all_clarifications) == 0:
            all_clarifications = [log_dict['orig_instruction']]
        all_clarifications = [x.strip() for x in all_clarifications if len(x.strip()) > 0]
        all_clarifications = [inst_transform(orig_inst, x) for x in all_clarifications][:MAX_CLARIFICATION]
        clarified_inputs = []
        for clarified_inst in all_clarifications:
            prompt_q = "Task: " + clarified_inst + '\nInput: ' + q + "\nOutput: "
            clarified_inputs.append(prompt_q)
    elif dataset_name == 'gsm8k':
        orig_q = log_dict['question']
        rewrite_qs = log_dict['self_clarification']
        if len(rewrite_qs) == 0:
            rewrite_qs = [orig_q]
        clarified_inputs = [prompt.strip() + '\n\nQuestion: ' + orig_q + '\n' for x in rewrite_qs]

    else:
        raise NotImplementedError
    return clarified_inputs

def post_process(model_response, dataset_name):
    if dataset_name == 'ambigqa':
        if len(model_response.split('\n')) >= 2:
            return model_response.split('\n')[0]
        else:
            return model_response
    elif dataset_name == 'ambig_inst':
        return model_response
    elif dataset_name == 'gsm8k':
        return model_response
    elif dataset_name == 'nq_open':
        return model_response


def main(args):
    dataset_name = args.dataset_name
    with open(args.clarification_path,'r',encoding='utf-8') as f:
        all_logs = json.load(f)
    system_prompt = load_system_prompt(dataset_name)
    fewshot_prompt = load_fewshot_prompt(dataset_name)
    best_n = SAMPLE_N

    output_path = args.output_path
    save_dir = os.path.dirname(output_path)
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    print("save logs to ", output_path)
    prog_bar = auto_tqdm(np.arange(len(all_logs)))

    all_res = []
    for q_idx in range(len(all_logs)):
        log_dict = all_logs[q_idx]
        copy_log_dict = copy.deepcopy(log_dict)

        clarified_inputs = format_query(log_dict, dataset_name, prompt = fewshot_prompt)
        copy_log_dict['clarified_all_ans'] = []
        for clarify_idx in range(len(clarified_inputs)):
            curr_input = clarified_inputs[clarify_idx]
            messages = format_message(system_prompt, curr_input)

            response = completion_with_backoff(
                model="gpt-3.5-turbo-0613",
                messages=messages,
                temperature=0.5,
                n = best_n,
                max_tokens=512,
                )
            model_outputs = [response['choices'][i]['message']['content'] for i in range(best_n)]
            processed = [post_process(x, dataset_name) for x in model_outputs]

            copy_log_dict['clarified_all_ans'].append(processed)

        all_res.append(copy_log_dict)
        prog_bar.update(1)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_res, f, indent = 4)

if __name__ == '__main__':
    main(args)
