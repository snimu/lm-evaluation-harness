export TASKS="truthfulqa,piqa,hellaswag,glue,lambada,arithmetic,mmlu"

lm-eval --model-name MoT --model_args "name=snimu/MoT_pad-lr_pull-nn_bpt-16_how-noop-noop_BTMdim-1024-1024-1024_niter-100000_773322" --num-fewshot 3 --batch_size="auto" --max_batch_size=512 --output_path="results/noop-noop-1024-1024-1024"
lm-eval --model-name MoT --model_args "name=snimu/MoT_pad-lr_pull-yn_bpt-16_how-concat-noop_BTMdim-48-256-1024_niter-100000_773322" --num-fewshot 3 --batch_size="auto" --max_batch_size=512 --output_path="results/concat-noop-48-256-1024"