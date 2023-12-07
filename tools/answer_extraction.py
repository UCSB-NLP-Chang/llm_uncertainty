import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
import json
import numpy as np
import copy
from typing import List
import tqdm
import argparse
from transformers import GPT2Tokenizer
from src.common import completion_with_backoff

INVLAID_RESUTL, EXCEED_LENGTH_LIMIT, SUCCESS = 1111, 2222, 3333

parser = argparse.ArgumentParser()
parser.add_argument("--log_path", type = str, required = True)
parser.add_argument("--prompt_path", type = str, required = True)
parser.add_argument("--answer_key", type = str, required = True)
parser.add_argument("--output_path", type = str, required = True)

args = parser.parse_args()


def extract_rewrite(model_ans: str):
    exts = []
    answer_sets = []
    lines = model_ans.split('\n')
    for line in lines:
        if line.startswith("Extraction"):
            try:
                ans = line.strip().split(":", 1)[1][1:]
                # ans = line.strip()[len("Extraction 1/N: "):].strip()
            except:
                print(f"{line} cannot be splitted!")
                ans = line.strip()[len("Extraction"):].strip()
            exts.append(ans)
        else:
            ans_set = line.strip()
            answer_sets.append(ans_set)
    return exts, answer_sets

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

def megabatch_forward(q: str, all_rewrite_answers: List[str]):
    ans_num_per_question = len(all_rewrite_answers[0])

    concate_all_rewrite_answers = []
    for ans_list in all_rewrite_answers:
        concate_all_rewrite_answers += ans_list

    ans2id = {}
    cleaned_answers = []
    for cdt_ans in concate_all_rewrite_answers:
        if cdt_ans.startswith('A: '):
            cdt_ans = cdt_ans[len('A: '):]
        cleaned_answers.append(cdt_ans)
        if cdt_ans not in ans2id: ans2id[cdt_ans] = len(ans2id)

    ext_rewrite_all_ans = []
    prompt_q = 'Q: ' + q

    for ans_id, cdt_ans in enumerate(list(ans2id.keys())):
        prompt_q = prompt_q + f'\nA{ans_id+1}: ' + cdt_ans
    prompt_q = prompt_q + '\nAnswer set at the begining: []'
    user_prompt = prompt_q

    tokens = tokenizer.tokenize(system_prompt + user_prompt)
    print(q)
    print(len(tokens))
    if len(tokens) > 2500:
        return EXCEED_LENGTH_LIMIT, None, None

    messages = format_message(system_prompt, user_prompt)

    response = completion_with_backoff(
        model="gpt-3.5-turbo-0613",
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        n=1,
        )
    ans_model = response['choices'][0]['message']['content']
    print(ans_model)

    ext_ans, ans_sets = extract_rewrite(ans_model)

    if len(ext_ans) != len(ans2id):
        print("failed!")
        print(ans_model)
        ext_rewrite_all_ans.append(cleaned_answers)
        return INVLAID_RESUTL, None, None

    map_back_list = []
    for ans_index, ans in enumerate(cleaned_answers):
        map_back_list.append(ext_ans[ans2id[ans]])
        if ans_index % ans_num_per_question == ans_num_per_question - 1:
            ext_rewrite_all_ans.append(map_back_list)
            map_back_list = []

    return SUCCESS, ext_rewrite_all_ans, ans_sets

def batch_forward(q: str, rewrite_answers, prev_ans_set):
    prompt_q = 'Q: ' + q

    ans2id = {}
    cleaned_answers = []
    for cdt_ans in rewrite_answers:
        if cdt_ans.startswith('A: '):
            cdt_ans = cdt_ans[len('A: '):]
        cleaned_answers.append(cdt_ans)
        if cdt_ans not in ans2id: ans2id[cdt_ans] = len(ans2id)

    for ans_id, cdt_ans in enumerate(list(ans2id.keys())):
        prompt_q = prompt_q + f'\nA{ans_id+1}: ' + cdt_ans
    prompt_q = prompt_q + f'\nAnswer set at the begining: {prev_ans_set}'
    user_prompt = prompt_q

    tokens = tokenizer.tokenize(system_prompt + user_prompt)
    print(q)
    print(len(tokens))

    _max_tokens = 2000
    curr_model = "gpt-3.5-turbo-0613"
    if len(tokens) > 2000:
        curr_model = "gpt-3.5-turbo-1106"


    print(f"Answer set at the begining: {prev_ans_set}")
    print(temperature)
    messages = format_message(system_prompt, user_prompt)

    response = completion_with_backoff(
        model=curr_model,
        messages=messages,
        temperature=temperature,
        max_tokens=_max_tokens,
        n=1,
        )
    ans_model = response['choices'][0]['message']['content']
    ext_ans, ans_sets = extract_rewrite(ans_model)

    if len(ext_ans) != len(ans2id):
        print("failed!")
        print(ans_model)

        return INVLAID_RESUTL, [], prev_ans_set

    map_back_list = []
    for ans in cleaned_answers:
        map_back_list.append(ext_ans[ans2id[ans]])
    return SUCCESS, map_back_list, ans_sets[-1]

if __name__ == '__main__':
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

    system_prompt = open(args.prompt_path,'r',encoding='utf-8').read().strip()

    output_path = args.output_path
    save_dir = os.path.dirname(output_path)
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    print("save logs to ", output_path)

    with open(args.log_path,'r',encoding='utf-8') as f:
        test_data = json.load(f)

    multiple_outputs = type(test_data[0][args.answer_key][0]) == list
    if multiple_outputs:
        print("There are more than one output distribution; Extract them together.")

    max_tokens=1500
    temperature=0
    sample_n = 1
    extract_ans_key = 'ext_' + args.answer_key


    all_results = []
    for idx in tqdm.tqdm(range(len(test_data))):
        case = test_data[idx]
        q = case['question']
        all_rewrite_answers = case[args.answer_key]

        result = copy.deepcopy(case)

        if multiple_outputs:
            FLAG, ext_rewrite_all_ans, ans_sets = megabatch_forward(q, all_rewrite_answers)
            if FLAG == SUCCESS:
                result[extract_ans_key] = ext_rewrite_all_ans
                result['ext_ans_sets'] = ans_sets
                all_results.append(result)
            elif FLAG == EXCEED_LENGTH_LIMIT or FLAG == INVLAID_RESUTL:
                ext_rewrite_all_ans = []
                ans_sets = []
                prev_ans_set = "[]"
                for rewrite_answers in all_rewrite_answers:
                    FLAG, curr_ext_res, curr_ans_set = batch_forward(q, rewrite_answers, prev_ans_set)
                    print(curr_ans_set)
                    if curr_ans_set.startswith('Final answer set: '):
                        processed_curr_ans_set = curr_ans_set.strip()[len('Final answer set: '):]
                    elif curr_ans_set.startswith("Updated answer set"):
                        processed_curr_ans_set = curr_ans_set.strip()[len('Updated answer set: '):]
                    else:
                        processed_curr_ans_set = curr_ans_set.strip()
                    print("answer set after process: ", processed_curr_ans_set)

                    if FLAG == INVLAID_RESUTL:
                        ext_rewrite_all_ans.append([])
                        ans_sets.append(None)
                    else:
                        ext_rewrite_all_ans.append(curr_ext_res)
                        prev_ans_set = processed_curr_ans_set
                        ans_sets.append(curr_ans_set)
                result[extract_ans_key] = ext_rewrite_all_ans
                result['ext_ans_sets'] = ans_sets
                all_results.append(result)

            else:
                raise NotImplementedError
        else:
            prev_ans_set = "[]"
            rewrite_answers = all_rewrite_answers
            FLAG, curr_ext_res, curr_ans_set = batch_forward(q, rewrite_answers, prev_ans_set)
            print(curr_ans_set)
            if curr_ans_set.startswith('Final answer set: '):
                processed_curr_ans_set = curr_ans_set.strip()[len('Final answer set: '):]
            elif curr_ans_set.startswith("Updated answer set"):
                processed_curr_ans_set = curr_ans_set.strip()[len('Updated answer set: '):]
            else:
                processed_curr_ans_set = curr_ans_set.strip()
            print("answer set after process: ", processed_curr_ans_set)

            if FLAG == INVLAID_RESUTL:
                ext_rewrite_all_ans = all_rewrite_answers
                ans_sets = None
            else:
                ext_rewrite_all_ans = curr_ext_res
                prev_ans_set = processed_curr_ans_set
                ans_sets = curr_ans_set
            result[extract_ans_key] = ext_rewrite_all_ans
            result['ext_ans_sets'] = ans_sets
            all_results.append(result)


    if output_path.endswith(".txt"):
        json_path = output_path.replace('.txt','.json')
    else:
        json_path = output_path
    with open(json_path, 'w', encoding = 'utf-8') as f:
        json.dump(all_results, f, indent = 4)
