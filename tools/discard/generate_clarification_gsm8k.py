import openai
import re
import os
import json
import datasets
import numpy as np
from tenacity import (
    retry,
    stop_after_attempt,
    wait_chain,
    wait_fixed
) 
from src.data_util import load_gsm_data

import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--paraphrase_prompt_path", type = str, required = True)
parser.add_argument("--output_path", type = str, required = True)
parser.add_argument("--sample", action = 'store_true')
parser.add_argument("--sample_n", type = int, required = True)

@retry(wait=wait_chain(*[wait_fixed(1) for i in range(3)] +
                       [wait_fixed(2) for i in range(2)] +
                       [wait_fixed(3)]))
def completion_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)

openai.api_key = "" # used my own key

def rephrase_format(question: str, rephrase_cot_prompt):
    if not question.startswith('Question: '):
        question = 'Question: ' + question
    prompt = rephrase_cot_prompt + question
    return prompt

def load_rephrase_prompt(args):
    rephrase_whole_prompts = open(args.paraphrase_prompt_path,'r',encoding='utf-8').read()
    _, rephrase_system_prompt, rephrase_cot_prompt = rephrase_whole_prompts.split("[--------------------------------------]",2)
    rephrase_system_prompt = rephrase_system_prompt.strip()
    rephrase_cot_prompt = rephrase_cot_prompt.strip()
    return rephrase_system_prompt, rephrase_cot_prompt

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


if __name__ == '__main__':
    args = parser.parse_args()
    output_path = args.output_path
    save_dir = os.path.dirname(output_path)
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    print("save logs to ", output_path)

    max_tokens=512
    if args.sample:
        temperature=1.0
        sample_n = args.sample_n
    else:
        temperature=0
        sample_n = 1

    rephrase_system_prompt, rephrase_cot_prompt = load_rephrase_prompt(args)

    train_data, test_data = load_gsm_data()
    test_data = test_data.select(np.arange(200))
    # test_data = test_data.select(np.arange(2))

    all_results = []
    for idx in tqdm.tqdm(range(len(test_data))):
        case = test_data[idx]
        q = case['question']
        a = case['answer']

        rewrite_list = []
        system_prompt = rephrase_system_prompt
        user_prompt = rephrase_format(q, rephrase_cot_prompt)

        messages = format_message(system_prompt, user_prompt)

        response = completion_with_backoff(
            model="gpt-3.5-turbo-0613",
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            n=sample_n,
            )
        ans_model_list = []
        for sample_id in range(sample_n):
            ans_model = response['choices'][sample_id]['message']['content']
            ans_model_list.append(ans_model)
        for model_output in ans_model_list:
            lines = model_output.split('\n')
            last_line = lines[-1]
            if last_line.startswith("Rephrase: "):
                last_line = last_line[len("Rephrase: "): ]
                rewrite_list.append(last_line)

        result = {
            'idx': idx,
            'question': q,
            'self_clarification': rewrite_list,
            'answer': a,
        }
        all_results.append(result)

    if output_path.endswith(".txt"):
        json_path = output_path.replace('.txt','.json')
    else:
        json_path = output_path
    with open(json_path, 'w', encoding = 'utf-8') as f:
        json.dump(all_results, f, indent = 4)

