lm-eval --model "causal-ul2" --model_args="size=1549,mode=r" --tasks="arithmetic,mmlu,truthfulqa" --batch_size="auto" --max_batch_size=256 --output_path="results-r-0shot/results.json" --log_samples --seed 1234 --num_fewshot=0
lm-eval --model "causal-ul2" --model_args="size=1549,mode=c" --tasks="arithmetic,mmlu,truthfulqa" --batch_size="auto" --max_batch_size=256 --output_path="results-c-0shot/results.json" --log_samples --seed 1234 --num_fewshot=0
lm-eval --model "causal-ul2" --model_args="size=1549,mode=r" --tasks="arithmetic,mmlu,truthfulqa" --batch_size="auto" --max_batch_size=256 --output_path="results-r-3shot/results.json" --log_samples --seed 2345 --num_fewshot=3
lm-eval --model "causal-ul2" --model_args="size=1549,mode=c" --tasks="arithmetic,mmlu,truthfulqa" --batch_size="auto" --max_batch_size=256 --output_path="results-c-3shot/results.json" --log_samples --seed 2345 --num_fewshot=3
lm-eval --model "causal-ul2" --model_args="size=1549,mode=r" --tasks="arithmetic,mmlu,truthfulqa" --batch_size="auto" --max_batch_size=256 --output_path="results-r-6shot/results.json" --log_samples --seed 3456 --num_fewshot=6
lm-eval --model "causal-ul2" --model_args="size=1549,mode=c" --tasks="arithmetic,mmlu,truthfulqa" --batch_size="auto" --max_batch_size=256 --output_path="results-c-6shot/results.json" --log_samples --seed 3456 --num_fewshot=6

# Previous tasks: --tasks="agieval_en,arithmetic,commonsense_qa,fda,glue,hellaswag,lambada,mmlu,truthfulqa"