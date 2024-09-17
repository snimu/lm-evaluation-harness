lm-eval --model causal-ul2 --tasks mmlu,hellaswag,mmlu_pro,agieval --batch_size auto --output_path eval-results/causal-ul2-C-fineweb10BT-240M-16heads-lr100.jsonl --model_args pretrained=snimu/causal-ul2-C-fineweb10BT-240M-16heads-lr100
lm-eval --model causal-ul2 --tasks mmlu,hellaswag,mmlu_pro,agieval --batch_size auto --output_path eval-results/causal-ul2-R-fineweb10BT-240M-16heads-lr090.jsonl --model_args pretrained=snimu/causal-ul2-R-fineweb10BT-240M-16heads-lr090
lm-eval --model causal-ul2 --tasks mmlu,hellaswag,mmlu_pro,agieval --batch_size auto --output_path eval-results/causal-ul2-C-fineweb10BT-773M-26heads-lr100.jsonl --model_args pretrained=snimu/causal-ul2-C-fineweb10BT-773M-26heads-lr100
lm-eval --model causal-ul2 --tasks mmlu,hellaswag,mmlu_pro,agieval --batch_size auto --output_path eval-results/causal-ul2-R-fineweb10BT-773M-26heads-lr090.jsonl --model_args pretrained=snimu/causal-ul2-R-fineweb10BT-773M-26heads-lr090