import openai
import re
import os
import json
import numpy as np
import copy
from tenacity import (
    retry,
    stop_after_attempt,
    wait_chain,
    wait_fixed
) 

import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--log_path", type = str, required = True)
parser.add_argument("--prompt_path", type = str, required = True)
parser.add_argument("--output_path", type = str, required = True)
parser.add_argument("--sample", action = 'store_true')
parser.add_argument("--sample_n", type = int, required = True)
parser.add_argument("--read_from_gt", action = 'store_true')

args = parser.parse_args()

@retry(wait=wait_chain(*[wait_fixed(1) for i in range(3)] +
                       [wait_fixed(2) for i in range(2)] +
                       [wait_fixed(3)]))
def completion_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)

def extract_rewrite(model_ans: str):
    lines = model_ans.split('\n')
    extract_list = []
    others = []
    for line in lines:
        # if line.startswith('Clarification'):
        #     extract_list.append(line[len("Clarification 1: "):])
        if line.startswith('Rephrase'):
            ext = line[len("Rephrase 1: "):]
            if 'specific' in ext.lower():
                continue
            extract_list.append(ext)
        elif line.startswith('Rephrased question'):
            ext = line[len("Rephrased question 1: "):]
            extract_list.append(ext)
        elif line.startswith('Rephrased'):
            ext = line[len("Rephrased 1: "):]
            extract_list.append(ext)
        elif line.startswith("Question: "):
            break
        else:
            others.append(line)
    return extract_list, '\n'.join(others)

whole_prompts = open(args.prompt_path,'r',encoding='utf-8').read()
commets, system_prompt, cot_prompt = whole_prompts.split("[--------------------------------------]",2)
system_prompt = system_prompt.strip()
cot_prompt = cot_prompt.strip()

output_path = args.output_path
save_dir = os.path.dirname(output_path)
if not os.path.exists(save_dir): os.makedirs(save_dir)
print("save logs to ", output_path)
openai.api_key = "" # used my own key

with open(args.log_path,'r',encoding='utf-8') as f:
    test_data = json.load(f)

max_tokens=512
if args.sample:
    temperature=0.5
    sample_n = args.sample_n
else:
    temperature=0
    sample_n = 1

# test_data = test_data[100:]

all_results = []
for idx in tqdm.tqdm(range(len(test_data))):
    case = test_data[idx]
    q = case['question']
    if args.read_from_gt:
        label = case['label']
        if "singleAnswer" in label:
            model_rephrase = []
        else:
            model_rephrase = case['gt_rewrite']
            case['self_clarification'] = model_rephrase
    elif 'self_clarification' not in case:
        model_rephrase = []
    else:
        model_rephrase = case['self_clarification']

    result = copy.deepcopy(case)
    if len(model_rephrase) == 0:

        prompt_q = 'Question: ' + q
        prompt_full = cot_prompt + '\n\n' + prompt_q


        if len(system_prompt) > 1:
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt_full},
            ]
        else:
            messages=[
                {"role": "user", "content": prompt_full},
            ]
        response = completion_with_backoff(
            model="gpt-3.5-turbo-0613",
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            n=sample_n,
            )
        ans_model_list = []
        other_outputs = []
        for sample_id in range(sample_n):
            ans_model = response['choices'][sample_id]['message']['content']
            extraction, others = extract_rewrite(ans_model)
            ans_model_list += extraction
            other_outputs.append(others)

        result['self_clarification'] = ans_model_list
    all_results.append(result)

if output_path.endswith(".txt"):
    json_path = output_path.replace('.txt','.json')
else:
    json_path = output_path
with open(json_path, 'w', encoding = 'utf-8') as f:
    json.dump(all_results, f, indent = 4)
