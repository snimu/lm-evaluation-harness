lm-eval --model "causal-ul2" --model_args="size=1549,mode=r" --tasks="arithmetic,mmlu,truthfulqa" --batch_size="auto" --max_batch_size=512 --output_path="results-r-0shot/results.json" --log_samples --seed 1234 --num_fewshot=0
lm-eval --model "causal-ul2" --model_args="size=1549,mode=c" --tasks="arithmetic,mmlu,truthfulqa" --batch_size="auto" --max_batch_size=512 --output_path="results-c-0shot/results.json" --log_samples --seed 1234 --num_fewshot=0
lm-eval --model "causal-ul2" --model_args="size=1549,mode=r" --tasks="arithmetic,mmlu,truthfulqa" --batch_size="auto" --max_batch_size=512 --output_path="results-r-3shot/results.json" --log_samples --seed 2345 --num_fewshot=3
lm-eval --model "causal-ul2" --model_args="size=1549,mode=c" --tasks="arithmetic,mmlu,truthfulqa" --batch_size="auto" --max_batch_size=512 --output_path="results-c-3shot/results.json" --log_samples --seed 2345 --num_fewshot=3
lm-eval --model "causal-ul2" --model_args="size=1549,mode=r" --tasks="arithmetic,mmlu,truthfulqa" --batch_size="auto" --max_batch_size=512 --output_path="results-r-5shot/results.json" --log_samples --seed 3456 --num_fewshot=5
lm-eval --model "causal-ul2" --model_args="size=1549,mode=c" --tasks="arithmetic,mmlu,truthfulqa" --batch_size="auto" --max_batch_size=512 --output_path="results-c-5shot/results.json" --log_samples --seed 3456 --num_fewshot=5

lm-eval --model "causal-ul2" --model_args="size=2556,mode=r,temperature=0.0" --tasks="truthfulqa,piqa,hellaswag,glue,lambada" --batch_size="auto" --max_batch_size=512 --output_path="results-r-3shot-temp0/results.json" --log_samples --seed 4567 --num_fewshot=3
lm-eval --model "causal-ul2" --model_args="size=2556,mode=c,temperature=0.0" --tasks="truthfulqa,piqa,hellaswag,glue,lambada" --batch_size="auto" --max_batch_size=512 --output_path="results-c-3shot-temp0/results.json" --log_samples --seed 4567 --num_fewshot=3

lm-eval --model "causal-ul2" --model_args="size=2556,mode=r,temperature=1.0" --tasks="truthfulqa,piqa,hellaswag,glue,lambada" --batch_size="auto" --max_batch_size=512 --output_path="results-r-3shot-temp1/results.json" --log_samples --seed 4567 --num_fewshot=3
lm-eval --model "causal-ul2" --model_args="size=2556,mode=c,temperature=1.0" --tasks="truthfulqa,piqa,hellaswag,glue,lambada" --batch_size="auto" --max_batch_size=512 --output_path="results-c-3shot-temp1/results.json" --log_samples --seed 4567 --num_fewshot=3

lm-eval --model "causal-ul2" --model_args="size=2556,mode=r,temperature=0.0,kth_token=2" --tasks="truthfulqa,piqa,hellaswag,glue,lambada" --batch_size="auto" --max_batch_size=512 --output_path="results-r-3shot-temp0-kth2/results.json" --log_samples --seed 4567 --num_fewshot=3
lm-eval --model "causal-ul2" --model_args="size=2556,mode=c,temperature=0.0,kth_token=2" --tasks="truthfulqa,piqa,hellaswag,glue,lambada" --batch_size="auto" --max_batch_size=512 --output_path="results-c-3shot-temp0-kth2/results.json" --log_samples --seed 4567 --num_fewshot=3

# Previous tasks: --tasks="agieval_en,arithmetic,commonsense_qa,fda,glue,hellaswag,lambada,mmlu,truthfulqa"