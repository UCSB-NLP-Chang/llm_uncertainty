python tools/generate_clarification.py \
       --dataset_name gsm8k  \
       --output_path logs/clarification/gsm8k.json \
       --sample --sample_n 2

python forward.py --dataset_name gsm8k \
       --clarification_path logs/clarification/gsm8k.json \
       --output_path logs/forward/gsm8k_forward.json

python evaluate_uq_gsm8k.py \
       --log_path logs/forward/gsm8k_forward.json \
       --output_path logs/uq_eval/gsm8k.json \
       --answer_key clarified_all_ans


python tools/compute_metrics_gsm8k.py
