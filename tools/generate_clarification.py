import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
import re
import json
import numpy as np
import copy
import tqdm
import argparse
from src.model_util import ICLModel
from src.prompt_util import load_clarification_system_prompt, load_clarification_user_prompt
from src.data_util import load_data
from src.common import completion_with_backoff

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", type = str, required = True)
parser.add_argument("--output_path", type = str, required = True)
parser.add_argument("--sample", action = 'store_true')
parser.add_argument("--sample_n", type = int, required = True)

args = parser.parse_args()


def extract_clarification_ambigqa(model_ans: str):
    lines = model_ans.split('\n')
    extract_list = []
    others = []
    pattern = r'^\d'
    for line in lines:
        if line.startswith('Clarifications'):
            continue
        match = re.match(pattern, line)
        if match:
            ext = line.strip()[len("1. "):]
            extract_list.append(ext)
        else:
            others.append(line.strip())
    return extract_list, others

def extract_clarification_ambiginst(model_output: str):
    if "No clarification needed" in model_output:
        return [],[]
    if not "Disambiguations:" in model_output:
        return [], []
    lines = model_output.split('Disambiguations:')[-1].strip().split('\n')
    all_exts = []
    others = []
    for line in lines:
        if len(line.strip()) == 0:
            break
        ext = line.strip()[len('1. '):]
        all_exts.append(ext)
    return all_exts, others

def extract_clarification_nq(model_ans: str):
    lines = model_ans.split('\n')
    extract_list = []
    others = []
    for line in lines:
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
    return extract_list, others

def extract_clarification_gsm8k(model_output: str):
    extract_list = []
    lines = model_output.split('\n')
    for line in lines:
        if line.startswith("Rephrase "):
            line = line[len("Rephrase 1: "): ]
            extract_list.append(line)
    return extract_list, []

def extract_clarification(model_output, dataset_name):
    if dataset_name == 'ambigqa':
        return extract_clarification_ambigqa(model_output)
    elif dataset_name == 'ambig_inst':
        return extract_clarification_ambiginst(model_output)
    elif dataset_name == 'nq_open':
        return extract_clarification_nq(model_output)
    elif dataset_name == 'gsm8k':
        return extract_clarification_gsm8k(model_output)


def load_db():
    pos_db_path = 'logs/dataset/ambigqa/ambigqa_train_ambig.json'
    neg_db_path = 'logs/dataset/ambigqa/ambigqa_train_unambig.json'
    with open(pos_db_path, 'r', encoding='utf-8') as f:
        pos_db = json.load(f)
    with open(neg_db_path, 'r', encoding='utf-8') as f:
        neg_db = json.load(f)
    return pos_db, neg_db

def format_query(log_dict, dataset_name, user_prompt, icl_selector):
    if dataset_name == 'ambigqa':
        q = log_dict['question']

        icl_examples = icl_selector.format_icl_prompt(question = q, ambig_num = 8, unambig_num = 8)
        icl_example_str = '\n\n'.join(icl_examples)

        prompt_q = 'Original Question: ' + q
        prompt_full = icl_example_str + '\n\n' + prompt_q
    elif dataset_name == 'nq_open':
        q = log_dict['question']
        prompt_q = 'Question: ' + q
        prompt_full = user_prompt + '\n\n' + prompt_q
    elif dataset_name == 'ambig_inst':
        orig_inst = log_dict['orig_instruction']
        input_q = log_dict['input']
        prompt_full = user_prompt + '\n\n'
        prompt_full += 'Original Task Instruction: ' + orig_inst + '\n'
        prompt_full += f'Input: ' +  input_q.strip() + '\n\n'
    elif dataset_name == 'gsm8k':
        q = log_dict['question']
        if not q.startswith('Question: '):
            q = 'Question: ' + q
        prompt_full = user_prompt + q

    return prompt_full


def main(args):
    system_prompt = load_clarification_system_prompt(args.dataset_name)
    user_prompt = load_clarification_user_prompt(args.dataset_name)

    if args.dataset_name == 'ambigqa':
        pos_db, neg_db = load_db()
        icl_selector = ICLModel(positive_db = pos_db, negative_db = neg_db)
    else:
        icl_selector = None

    output_path = args.output_path
    save_dir = os.path.dirname(output_path)
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    print("save logs to ", output_path)
    model_index = 'gpt-4' if args.dataset_name == 'ambigqa' else 'gpt-3.5-turbo-0613'

    test_data = load_data(args.dataset_name)

    all_results = []
    for idx in tqdm.tqdm(range(len(test_data))):
        case = test_data[idx]
        prompt_full = format_query(case, args.dataset_name, user_prompt, icl_selector)

        max_tokens=512
        if args.sample:
            temperature=1.0
            sample_n = args.sample_n
        else:
            temperature=0
            sample_n = 1

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
            model=model_index,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            n=sample_n,
            )
        ans_model_list = []
        other_outputs = []
        for sample_id in range(sample_n):
            ans_model = response['choices'][sample_id]['message']['content']
            extraction, others = extract_clarification(ans_model, args.dataset_name)
            ans_model_list += extraction
            other_outputs.append(others)

        result = copy.deepcopy(case)
        if args.dataset_name == 'ambig_inst':
            result['orig_inst'] = case['orig_instruction']            
        result['self_clarification'] = ans_model_list
        result['others'] = other_outputs
        all_results.append(result)


    if output_path.endswith(".txt"):
        json_path = output_path.replace('.txt','.json')
    else:
        json_path = output_path
    with open(json_path, 'w', encoding = 'utf-8') as f:
        json.dump(all_results, f, indent = 4)

if __name__ == '__main__':
    main(args)
