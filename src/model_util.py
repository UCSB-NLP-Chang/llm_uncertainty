import torch
import numpy as np
from typing import List
import random
from sentence_transformers import SentenceTransformer, util
from .prompt_util import clarify_format, example_clarify

class ICLModel():
    def __init__(self, positive_db: List[dict], negative_db: List[dict]) -> None:
        self.positive_db = positive_db  ## pos = ambig
        self.negative_db = negative_db

        self.model = SentenceTransformer('all-mpnet-base-v2', cache_folder = './model_cache/sentence_transformer/all-mpnet-base-v2', device = torch.device('cuda'))

        self.positive_embeddings = self.encode_db(self.positive_db)
        self.negative_embeddings = self.encode_db(self.negative_db)


    def encode_db(self, db, max_bsz = 32):
        questions = [x['question'] for x in db]
        db_embeddings = []
        num_batches = len(questions) // max_bsz
        for idx in range(num_batches):
            curr_batch = questions[idx * max_bsz: (idx + 1) * max_bsz]
            curr_embeddings = self.model.encode(curr_batch, convert_to_numpy = False, convert_to_tensor =True)
            db_embeddings.append(curr_embeddings)
        if num_batches * max_bsz < len(questions):
            curr_batch = questions[num_batches * max_bsz:]
            curr_embeddings = self.model.encode(curr_batch, convert_to_numpy = False, convert_to_tensor =True)
            db_embeddings.append(curr_embeddings)
        db_embeddings = torch.cat(db_embeddings, axis = 0)
        print(db_embeddings.size())
        return db_embeddings

    def search_examples(self, question, ambig_num, unambig_num):
        if type(question) == str:
            question_embedding = self.model.encode([question], convert_to_numpy = False)
        else:
            question_embedding = self.model.encode(question, convert_to_numpy = False)
        
        pos_results = util.semantic_search(question_embedding, self.positive_embeddings, top_k = ambig_num)[0]
        neg_results = util.semantic_search(question_embedding, self.negative_embeddings, top_k = unambig_num)[0]
        return pos_results, neg_results

    def deduplicate(self, items):
        filtered = []
        for item in items:
            if item not in filtered:
                filtered.append(item)
        return filtered

    def format_icl_prompt(self, question, ambig_num, unambig_num):
        pos_results, neg_results = self.search_examples(question=question, ambig_num=ambig_num, unambig_num=unambig_num)
        pos_idxs = [x['corpus_id'] for x in pos_results]
        neg_idxs = [x['corpus_id'] for x in neg_results]
        pos_icl_examples = []
        neg_icl_examples = []
        for idx in pos_idxs:
            example = self.positive_db[idx]
            q = example['question']
            rewrites = example['gt_rewrite']
            rewrites = self.deduplicate(rewrites)
            all_formated_clarifies = []
            for rewrite_idx, rewrite in enumerate(rewrites):
                clarify_exp = clarify_format.format(idx = rewrite_idx + 1, clarification = rewrite)
                all_formated_clarifies.append(clarify_exp)
            all_formated_clarifies = '\n'.join(all_formated_clarifies)
            icl_exp = example_clarify.format(orig_question = q, all_clarifications = all_formated_clarifies)
            pos_icl_examples.append(icl_exp)

        for idx in neg_idxs:
            example = self.negative_db[idx]
            q = example['question']
            all_formated_clarifies = 'No clarification needed.'
            icl_exp = example_clarify.format(orig_question = q, all_clarifications = all_formated_clarifies)
            neg_icl_examples.append(icl_exp)

        all_icl_examples = pos_icl_examples + neg_icl_examples
        random.shuffle(all_icl_examples)
        return all_icl_examples
