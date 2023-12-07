python tools/generate_clarification.py \
       --dataset_name nq_open  \
       --output_path logs/clarification/nq.json \
       --sample --sample_n 2

python forward.py --dataset_name nq_open \
       --clarification_path logs/clarification/nq.json \
       --output_path logs/forward/nq_forward.json


python tools/answer_extraction.py \
       --log_path logs/forward/nq_forward.json \
       --prompt_path lib_prompt/common/answer_extraction.txt \
       --answer_key clarified_all_ans \
       --output_path logs/forward/nq_forward_ext.json


python evaluate_uq_qa.py \
       --log_path logs/forward/nq_forward_ext.json \
       --output_path logs/uq_eval/nq.json \
       --answer_key ext_clarified_all_ans

python tools/evaluate_correctness.py \
       --log_path logs/uq_eval/nq.json \
       --prompt_path lib_prompt/evaluation/nq_eval.txt \
       --output_path logs/uq_eval/gpt_eval_nq.json

python tools/compute_metrics_nq.py
