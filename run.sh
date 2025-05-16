export TASKS="truthfulqa,piqa,hellaswag,glue,lambada,arithmetic,mmlu"

lm-eval --model MoT --model_args "name=snimu/MoT_pad-lr_pull-nn_bpt-16_how-noop-noop_BTMdim-1024-1024-1024_niter-100000_773322" --num_fewshot 3 --batch_size="auto" --max_batch_size=512 --output_path="results/noop-noop-1024-1024-1024-greedy" --tasks $TASKS
lm-eval --model MoT --model_args "name=snimu/MoT_pad-lr_pull-yn_bpt-16_how-concat-noop_BTMdim-48-256-1024_niter-100000_773322" --num_fewshot 3 --batch_size="auto" --max_batch_size=512 --output_path="results/concat-noop-48-256-1024-greedy" --tasks $TASKS

lm-eval --model MoT --model_args "name=snimu/MoT_pad-lr_pull-nn_bpt-16_how-noop-noop_BTMdim-1024-1024-1024_niter-100000_773322,temperature=0.5" --num_fewshot 3 --batch_size="auto" --max_batch_size=512 --output_path="results/noop-noop-1024-1024-1024-temp-050" --tasks $TASKS
lm-eval --model MoT --model_args "name=snimu/MoT_pad-lr_pull-yn_bpt-16_how-concat-noop_BTMdim-48-256-1024_niter-100000_773322,temperature=0.5" --num_fewshot 3 --batch_size="auto" --max_batch_size=512 --output_path="results/concat-noop-48-256-1024-temp-050" --tasks $TASKS

lm-eval --model MoT --model_args "name=snimu/MoT_pad-lr_pull-nn_bpt-16_how-noop-noop_BTMdim-1024-1024-1024_niter-100000_773322,temperature=1.0" --num_fewshot 3 --batch_size="auto" --max_batch_size=512 --output_path="results/noop-noop-1024-1024-1024-temp-100" --tasks $TASKS
lm-eval --model MoT --model_args "name=snimu/MoT_pad-lr_pull-yn_bpt-16_how-concat-noop_BTMdim-48-256-1024_niter-100000_773322,temperature=1.0" --num_fewshot 3 --batch_size="auto" --max_batch_size=512 --output_path="results/concat-noop-48-256-1024-temp-100" --tasks $TASKS
