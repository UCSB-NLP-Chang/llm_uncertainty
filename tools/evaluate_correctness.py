import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from tqdm.auto import tqdm as auto_tqdm
import copy
import json
import numpy as np
import argparse
from src.common import completion_with_backoff

parser = argparse.ArgumentParser()
parser.add_argument("--log_path", type = str, required = True)
parser.add_argument("--prompt_path", type = str, required = True)
parser.add_argument("--output_path", type = str, required = True)
args = parser.parse_args()


with open(args.log_path, 'r', encoding='utf-8') as f:
    all_logs = json.load(f)

fewshot_prompt = open(args.prompt_path,'r',encoding='utf-8').read().strip()
output_path = args.output_path
save_dir = os.path.dirname(output_path)
if not os.path.exists(save_dir): os.makedirs(save_dir)
print("save logs to ", output_path)

model_ans_name = 'most_freq_ans'

prog_bar = auto_tqdm(np.arange(len(all_logs)))
all_res = []

num_query = 0

for q_idx in range(len(all_logs)):
    log_dict = all_logs[q_idx]
    copy_log_dict = copy.deepcopy(log_dict)

    orig_q = log_dict['question']
    ref_answers = log_dict['answer']
    pred_answer = log_dict[model_ans_name]
    ans2rating = {}

    scores = []
    for ref_answer in ref_answers:
        user_prompt = fewshot_prompt + '\n\nQuestion: ' + orig_q.strip() + '\nReference: ' + ref_answer.strip() + '\nAnswer: ' + pred_answer + '\nRating: '
        print(user_prompt)
        pause = input("??")
        response = completion_with_backoff(
            model="gpt-3.5-turbo-0613",
            messages=[
                {"role": "user", "content": user_prompt},
            ],
            temperature=0, n = 1, max_tokens=512,
        )
        score = response['choices'][0]['message']['content']
        scores.append(score)

    copy_log_dict['gptscore'] = scores
    all_res.append(copy_log_dict)
    prog_bar.update(1)

with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(all_res, f, indent = 4)

