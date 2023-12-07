import openai
import re
import os
import json
import copy
import random
from tenacity import retry, wait_chain, wait_fixed
import tqdm


openai.api_key = "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" # used my own key
@retry(wait=wait_chain(*[wait_fixed(1) for i in range(3)] +
                       [wait_fixed(2) for i in range(2)] +
                       [wait_fixed(3)]))
def completion_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)

def extract_clarification(model_output: str):
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

def load_data():
    with open("logs/synthesized_data/data.json",'r',encoding='utf-8') as f:
        data = json.load(f)
    return data

best_n = 1
temperature = 0

if __name__ == '__main__':
    test_data = load_data()
    prompt_path = 'lib_prompt/common/clarification_default.txt'
    prompt = open(prompt_path,'r').read().strip()

    clarify_logs = []

    cached = []
    for data_dict in tqdm.tqdm(test_data):
        orig_inst = data_dict['orig_instruction']
        if 'program' in orig_inst:
            continue
        input_q = data_dict['input']
        user_prompt = prompt + '\n\n'
        user_prompt += 'Original Task Instruction: ' + orig_inst + '\n'
        user_prompt += f'Input: ' +  input_q.strip() + '\n\n'

        # print(user_prompt)
        # pause = input("???")

        response = completion_with_backoff(
            model="gpt-3.5-turbo-0613",
            # model="gpt-4-1106-preview",
            # model="gpt-4",
            messages=[
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            n = best_n,
            max_tokens=512,
            )


        all_exts = []
        all_others = []
        # print(orig_inst)
        # print(input_q)
        for idx in range(best_n):
            ans = response['choices'][idx]['message']['content']
            # print(ans)
            # print('\n-------------------\n')
            exts, others = extract_clarification(ans)
            all_exts += exts
            all_others += others

        log_dict = copy.deepcopy(data_dict)
        log_dict['orig_inst'] = orig_inst
        log_dict['self_clarification'] = all_exts

        clarify_logs.append(log_dict)

    output_path = f'logs/clarification/self_clarified.json'
    if not os.path.exists(os.path.dirname(output_path)): os.makedirs(os.path.dirname(output_path))
    with open(output_path,'w',encoding='utf-8') as f:
        json.dump(clarify_logs, f, indent = 4)


