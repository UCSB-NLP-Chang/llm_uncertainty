import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
import datasets
import numpy as np
import json
import random

def prepare_ambigqa(per_class_size = 100):
    test_data = datasets.load_dataset('ambig_qa', 'light', split='validation', cache_dir = 'dataset_cache')

    ambig_examples = []
    unambig_examples = []

    for idx in range(len(test_data)):
        data = test_data[idx]
        question = data['question']
        annotate_dict = data['annotations']
        label = annotate_dict['type']

        if 'singleAnswer' in label and 'multipleQAs' in label:
            continue

        num_annotate = len(annotate_dict['qaPairs'])
        gt_rewrite = []
        gt_answers = []
        for anno_idx in range(num_annotate):
            curr_label = label[anno_idx]
            curr_single_answers = annotate_dict['answer'][anno_idx]
            if curr_label == 'singleAnswer':
                gt_rewrite += [question]
                gt_answers += [curr_single_answers]
            elif curr_label == 'multipleQAs':
                gt_rewrite += annotate_dict['qaPairs'][anno_idx]['question']
                gt_answers += annotate_dict['qaPairs'][anno_idx]['answer']

        log_dict = {
            'question': question,
            'label': label,
            'orig_annotation': annotate_dict,
            'gt_rewrite': gt_rewrite,
            'answer': gt_answers,
        }
        if 'singleAnswer' in label:
            unambig_examples.append(log_dict)
        else:
            ambig_examples.append(log_dict)

    np.random.seed(666)
    random.seed(666)

    unambig_rand_idxs = np.random.choice(len(unambig_examples), len(unambig_examples), replace = False)
    ambig_rand_idxs = np.random.choice(len(ambig_examples), len(ambig_examples), replace = False)

    selected_unambig_list = []
    selected_ambig_list = []
    for idx in unambig_rand_idxs[:per_class_size]:
        selected_unambig_list.append(unambig_examples[idx])

    for idx in ambig_rand_idxs[:per_class_size]:
        selected_ambig_list.append(ambig_examples[idx])

    final_list = selected_unambig_list + selected_ambig_list

    output_path = 'logs/dataset/ambigqa/ambigqa_dev_balance.json'
    save_dir = os.path.dirname(output_path)
    if not os.path.exists(save_dir): os.makedirs(save_dir)

    with open(output_path,'w',encoding='utf-8') as f:
        json.dump(final_list, f, indent = 4)

def extract_candidate_data():
    train_data = datasets.load_dataset('ambig_qa', 'light', split='train', cache_dir = 'dataset_cache')

    ambig_examples = []
    unambig_examples = []

    for idx in range(len(train_data)):
        data = train_data[idx]
        question = data['question']
        annotate_dict = data['annotations']
        label = annotate_dict['type']

        if 'singleAnswer' in label and 'multipleQAs' in label:
            continue

        num_annotate = len(annotate_dict['qaPairs'])
        gt_rewrite = []
        gt_answers = []
        for anno_idx in range(num_annotate):
            curr_label = label[anno_idx]
            curr_single_answers = annotate_dict['answer'][anno_idx]
            if curr_label == 'singleAnswer':
                gt_rewrite += [question]
                gt_answers += [curr_single_answers]
            elif curr_label == 'multipleQAs':
                gt_rewrite += annotate_dict['qaPairs'][anno_idx]['question']
                gt_answers += annotate_dict['qaPairs'][anno_idx]['answer']

        log_dict = {
            'question': question,
            'label': label,
            'orig_annotation': annotate_dict,
            'gt_rewrite': gt_rewrite,
            'answer': gt_answers,
        }
        if 'singleAnswer' in label:
            unambig_examples.append(log_dict)
        else:
            ambig_examples.append(log_dict)

    np.random.seed(666)
    random.seed(666)

    unambig_rand_idxs = np.random.choice(len(unambig_examples), len(unambig_examples), replace = False)
    ambig_rand_idxs = np.random.choice(len(ambig_examples), len(ambig_examples), replace = False)

    selected_unambig_list = []
    selected_ambig_list = []
    for idx in unambig_rand_idxs:
        selected_unambig_list.append(unambig_examples[idx])

    for idx in ambig_rand_idxs:
        selected_ambig_list.append(ambig_examples[idx])

    output_path = 'logs/dataset/ambigqa/ambigqa_train_ambig.json'
    save_dir = os.path.dirname(output_path)
    if not os.path.exists(save_dir): os.makedirs(save_dir)

    with open(output_path,'w',encoding='utf-8') as f:
        json.dump(selected_ambig_list, f, indent = 4)

    output_path = 'logs/dataset/ambigqa/ambigqa_train_unambig.json'
    save_dir = os.path.dirname(output_path)
    if not os.path.exists(save_dir): os.makedirs(save_dir)

    with open(output_path,'w',encoding='utf-8') as f:
        json.dump(selected_unambig_list, f, indent = 4)

if __name__ == '__main__':
    prepare_ambigqa()
    extract_candidate_data()

