import openai
import re
import os
import json
import numpy as np
import copy
from tenacity import retry, stop_after_attempt, wait_chain, wait_fixed
import tqdm
import argparse
from src.model_util import ICLModel

parser = argparse.ArgumentParser()
parser.add_argument("--log_path", type = str, required = True)
parser.add_argument("--prompt_path", type = str, required = True)
parser.add_argument("--output_path", type = str, required = True)
parser.add_argument("--nshot_ambig", type = int, required = True)
parser.add_argument("--nshot_unambig", type = int, required = True)
parser.add_argument("--sample", action = 'store_true')
parser.add_argument("--sample_n", type = int, required = True)

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
    return extract_list, '\n'.join(others)

def load_db():
    pos_db_path = 'logs/dataset/processed/amb_qa_train_ambig.json'
    neg_db_path = 'logs/dataset/processed/amb_qa_train_unambig.json'
    with open(pos_db_path, 'r', encoding='utf-8') as f:
        pos_db = json.load(f)
    with open(neg_db_path, 'r', encoding='utf-8') as f:
        neg_db = json.load(f)
    return pos_db, neg_db

def main(args):
    whole_prompts = open(args.prompt_path,'r',encoding='utf-8').read()
    commets, system_prompt, cot_prompt = whole_prompts.split("[--------------------------------------]",2)
    system_prompt = system_prompt.strip()

    pos_db, neg_db = load_db()
    icl_selector = ICLModel(positive_db = pos_db, negative_db = neg_db)
    

    output_path = args.output_path
    save_dir = os.path.dirname(output_path)
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    print("save logs to ", output_path)
    openai.api_key = "" # used my own key

    with open(args.log_path,'r',encoding='utf-8') as f:
        test_data = json.load(f)

    # test_data = test_data[:100]
    # test_data = test_data[100:200]

    all_results = []
    for idx in tqdm.tqdm(range(len(test_data))):
        case = test_data[idx]
        q = case['question']

        icl_examples = icl_selector.format_icl_prompt(question = q, ambig_num = args.nshot_ambig, unambig_num = args.nshot_unambig)
        icl_example_str = '\n\n'.join(icl_examples)

        prompt_q = 'Original Question: ' + q
        prompt_full = icl_example_str + '\n\n' + prompt_q

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
            model="gpt-4",
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

        result = copy.deepcopy(case)
        result['self_clarification'] = ans_model_list
        result['others'] = other_outputs
        all_results.append(result)

        if not prompt_q.endswith('\n'):
            prompt_q += '\n'


    if output_path.endswith(".txt"):
        json_path = output_path.replace('.txt','.json')
    else:
        json_path = output_path
    with open(json_path, 'w', encoding = 'utf-8') as f:
        json.dump(all_results, f, indent = 4)

if __name__ == '__main__':
    main(args)
