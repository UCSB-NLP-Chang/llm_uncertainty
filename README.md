## Uncertainty Quantification for LLMs

This is the official implementation for the paper, "Decomposing Uncertainty for Large Language Models through Input Clarification Ensembling".

### Requirements

The dependency packages can be found in `requirements.txt` file. One can use `pip install -r requirements.txt` to configure the environment. We use python 3.8 to run the experiments.


### Prepare the Data

Run the following script to prepare the data. 
```sh
python tools/prepare_data.py 
```

### Running the experiments
The overall pipeline is generate clarification $\rightarrow$ collect model answers $\rightarrow$ quantify uncertainty $\rightarrow$ evaluate the performance (either mistake detection or ambiguity detection). Before running experiments, you need to configure your OpenAI API key in `src/common.py` file.

1. Use the following script to generate clarifications

```sh
python tools/generate_clarification.py --dataset_name ambigqa  --output_path logs/clarification/ambigqa.json --sample --sample_n 2
```

`dataset_name`: choices include `nq_open, gsm8km, ambigqa, ambig_inst`

2. Then collect the model outputs based on the generated clarifications:

```sh
python forward.py --dataset_name ambigqa --clarification_path logs/clarification/ambigqa.json --output_path logs/forward/ambigqa_forward.json
```

3. (For Natural Question and AmbigQA) The uncertainty quantification requires the model output distribution. We sample multiple answers in Step 2 above and then count the answer frequencies to estimate the output distribution. For the GSM8K and synthetic AmbigInst dataset, we ignore the intermediate reasoning steps and take the final answer (*e.g.*, a real number from GSM8K) for frequency computation.

   However, for tasks such as Natural Question and AmbigQA, the ChatGPT model will often output a complete sentence rather than just a word or phrase as the answer. For example, given the question `When did the world’s population reach 7 billion?`, ChatGPT may generate several different answers such as `December 2017` and `The world’s  population reached 7 billion in December 2017`. Regarding them as two different answers can lead to an overestimation of the entropy of output distribution. Therefore, we follow previous work \[[1](https://github.com/lorenzkuhn/semantic_uncertainty/tree/main)\] and \[[2](https://github.com/zlin7/UQ-NLG/tree/main)\] to map these answers into different groups and each group contains semantically equaivalent answers. The output distribution is then computed on top of these groups.

   We empirically find relying the LLM itself to cluster the generated answers brings better performance compared to using an NLI model in  \[[1](https://github.com/lorenzkuhn/semantic_uncertainty/tree/main)\] and \[[2](https://github.com/zlin7/UQ-NLG/tree/main)\]. Use the following script to run our method:

 ```sh
 python tools/answer_extraction.py --log_path logs/forward/ambigqa_forward.json --prompt_path lib_prompt/common/answer_extraction.txt --answer_key clarified_all_ans --output_path logs/forward/ambigqa_forward_ext.json
 ```

   The above script will read the original answers stored in the logs under the key `clarified_all_ans`. The extracted answers will be then stored with the key `ext_clarified_all_ans` for next-step evaluation

4. Uncertainty quantification. Use the following script to quantify the uncertainty of model prediction:

```sh
python evaluate_uq_qa.py --log_path logs/forward/ambigqa_forward_ext.json --output_path logs/uq_eval/ambigqa.json --answer_key ext_clarified_all_ans
```
(As we have executed step 3, here we need to change the `answer_key` to `ext_clarified_all_ans`)

5. Performance evaluation. You can use the evaluation scripts under the `tools/` directory, such as
```sh
python tools/compute_metrics_ambigqa.py
```

Note: for the experiment on Natual Question dataset where we use the uncertainty to predict whether the model's answer is correct, we follow the experiment setting in \[[2](https://github.com/zlin7/UQ-NLG/tree/main)\] and use ChatGPT to judge the correctness of model's answer. Therefore, to evaluate the performance on Natural Question dataset, you need to run the following script before `python compute_metrics_nq.py`:

```sh
python evaluate_correctness.py --log_path logs/uq_eval/nq.json --prompt_path lib_prompt/evaluation/nq_eval.txt --output_path logs/uq_eval/gpt_eval_nq.json
```

You can find the (more detailed) running scripts for each dataset under the `scripts` directory.

Note: Currently, we have not integrated the running scripts for different datasets into a single one. We will update the code later on.



Some of our implementation of the experiments on GSM8K dataset comes from [chain-of-thought-hub](https://github.com/FranxYao/chain-of-thought-hub).

```
@article{fu2023chain,
  title={Chain-of-Thought Hub: A Continuous Effort to Measure Large Language Models' Reasoning Performance},
  author={Fu, Yao and Ou, Litu and Chen, Mingyu and Wan, Yuhao and Peng, Hao and Khot, Tushar},
  journal={arXiv preprint arXiv:2305.17306},
  year={2023}
}
```

The experiments on Natural Question follows the evaluation setting from [Semantic Uncertainty](https://github.com/lorenzkuhn/semantic_uncertainty/tree/main) and [UQ-NLG](https://github.com/zlin7/UQ-NLG/tree/main). Some of our implementation also comes from their repositories.

```
@inproceedings{kuhn2022semantic,
  title={Semantic Uncertainty: Linguistic Invariances for Uncertainty Estimation in Natural Language Generation},
  author={Kuhn, Lorenz and Gal, Yarin and Farquhar, Sebastian},
  booktitle={The Eleventh International Conference on Learning Representations},
  year={2022}
}
```

```
@article{lin2023generating,
  title={Generating with Confidence: Uncertainty Quantification for Black-box Large Language Models},
  author={Lin, Zhen and Trivedi, Shubhendu and Sun, Jimeng},
  journal={arXiv preprint arXiv:2305.19187},
  year={2023}
}
```

### Citation

```
@article{hou2023decomposing,
  title={Decomposing Uncertainty for Large Language Models through Input Clarification Ensembling},
  author={Hou, Bairu and Liu, Yujian and Qian, Kaizhi and Andreas, Jacob and Chang, Shiyu and Zhang, Yang},
  journal={arXiv preprint arXiv:2311.08718},
  year={2023}
}
```