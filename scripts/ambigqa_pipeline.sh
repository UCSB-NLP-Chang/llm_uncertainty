python tools/generate_clarification.py \
       --dataset_name ambigqa  \
       --output_path logs/clarification/ambigqa.json \
       --sample --sample_n 2

python forward.py --dataset_name ambigqa \
       --clarification_path logs/clarification/ambigqa.json \
       --output_path logs/forward/ambigqa_forward.json


python tools/answer_extraction.py \
       --log_path logs/forward/ambigqa_forward.json \
       --prompt_path lib_prompt/common/answer_extraction.txt \
       --answer_key clarified_all_ans \
       --output_path logs/forward/ambigqa_forward_ext.json


python evaluate_uq_qa.py \
       --log_path logs/forward/ambigqa_forward_ext.json \
       --output_path logs/uq_eval/ambigqa.json \
       --answer_key ext_clarified_all_ans

python tools/compute_metrics_ambigqa.py
