import regex
import re
import string
CONTRADICT, NEUTRAL, AGREE = 0, 1, 2

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

def normalize_answer(s):
    def remove_articles(text):
        return regex.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        if 'regard' in text:
            text = regex.sub(r'\([^)]*\)', '', text).strip()
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def recursive_normalize(object):
    if type(object) == list:
        new_obj = []
        for item in object:
            new_obj.append(recursive_normalize(item))
        return new_obj
    elif type(object) == str:
        # new_obj = object.translate(translator)
        new_obj = normalize_answer(object)
        return new_obj
    else:
        print(object)
        print(type(object))
        raise NotImplementedError


def exact_match_score(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)

def ems(prediction, ground_truths):
    return max([exact_match_score(prediction, gt) for gt in ground_truths])

def span_exact_match_score(prediction, ground_truth):
    norm_pred = normalize_answer(prediction)
    norm_gt = normalize_answer(ground_truth)
    return norm_gt in norm_pred

def span_ems(prediction, ground_truths):
    return max([span_exact_match_score(prediction, gt) for gt in ground_truths])

def is_ambig(label_list):
    if 'singleAnswer' not in label_list:
        return True
    else:
        return False


