python tools/generate_clarification.py \
       --dataset_name ambig_inst  \
       --output_path logs/clarification/ambig_inst.json \
       --sample --sample_n 2

python forward.py --dataset_name ambig_inst \
       --clarification_path logs/clarification/ambig_inst.json \
       --output_path logs/forward/ambig_inst_forward.json

python evaluate_uq_ambiginst.py \
       --log_path logs/forward/ambig_inst_forward.json \
       --output_path logs/uq_eval/ambig_inst.json \
       --answer_key clarified_all_ans

python tools/compute_metrics_ambiginst.py
