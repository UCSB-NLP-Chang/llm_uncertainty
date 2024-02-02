import string
import re
import openai
from tenacity import retry, wait_chain, wait_fixed
openai.api_key = "sk-" # used my own key

@retry(wait=wait_chain(*[wait_fixed(1) for i in range(3)] +
                       [wait_fixed(2) for i in range(2)] +
                       [wait_fixed(3)]))
def completion_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)


def majority_vote(answers):
    ans2freq = {}
    max_freq = 0
    max_ans = None
    for ans in answers:
        if ans not in ans2freq:
            ans2freq[ans] = 1
        else:
            ans2freq[ans] += 1
        if ans2freq[ans] > max_freq:
            max_ans = ans
            max_freq = ans2freq[ans]
    return max_ans, max_freq

def remove_punctuation(object):
    translator = str.maketrans('', '', string.punctuation)
    if type(object) == list:
        new_obj = []
        for item in object:
            new_obj.append(remove_punctuation(item))
        return new_obj
    elif type(object) == str:
        new_obj = object.strip().translate(translator).lower()
        return new_obj
    else:
        raise NotImplementedError


unk_word_pool = ['unknown', "I don't", "did not", "Not specified", "cannot be determined", "No Answer", "No final answer","I do not", "N/A", "No information", "It depends on", "I cannot","I can't ", "I am unable","don't know", "No answer", "Nobody", "enough information", "specific information", "There is no", "No specific","not provided", "None", "No character", "did not", "No output","Cannot answer", "Unavailable", "TBD", "To Be Determined", "I am asking about", "current year", "No translation", "depends on", "has not", 'unclear', "confusion", "incorrect", "not aware of", "invalid",'no one']

def check_answers(ans_list):
    unk_words = [x.lower() for x in unk_word_pool]
    purified_ans = []
    num_unk = 0
    for ans in ans_list:
        flag = False
        if ans.lower() in unk_words:
            flag = True
        for unk_w in unk_word_pool:
            if unk_w.lower() in ans.lower():
                flag = True
                break
        if flag:
            num_unk += 1
            purified_ans.append('unknown')
        else:
            purified_ans.append(ans)

    return purified_ans

def gsm8k_extract_ans(pred_str):
    ANS_RE = re.compile(r"\$?([0-9,]+)\.?\d*%?")
    pred = re.findall(ANS_RE, pred_str)
    if(len(pred) >= 1):
        pred = pred[-1]
        pred = pred.replace(",", "").replace(" ", "")
        try:
            return int(pred)
        except:
            return -1
    else:
        return -1

def ambiginst_extract_ans(model_ans):
    if "answer is" in model_ans:
        ext = model_ans.split('answer is')[-1].strip()
        return ext
    return model_ans


