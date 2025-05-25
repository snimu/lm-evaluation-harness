export TASKS="truthfulqa,piqa,hellaswag,glue,lambada,arithmetic,mmlu"

# TOKENS OUT
lm-eval --model MoT --model_args "name=snimu/MoT_pad-lr_pull-nn_bpt-16_how-noop-noop_BTMdim-1024-1024-1024_niter-100000_773322" --num_fewshot 3 --batch_size="auto" --max_batch_size=512 --output_path="results/noop-noop-1024-1024-1024-greedy" --tasks $TASKS
lm-eval --model MoT --model_args "name=snimu/MoT_pad-lr_pull-yn_bpt-16_how-concat-noop_BTMdim-48-256-1024_niter-100000_773322" --num_fewshot 3 --batch_size="auto" --max_batch_size=512 --output_path="results/concat-noop-48-256-1024-greedy" --tasks $TASKS
lm-eval --model MoT --model_args "name=snimu/MoT_pad-lr_pull-yn_bpt-16_how-concat-noop_BTMdim-64-768-1024_niter-100000_773322" --num_fewshot 3 --batch_size="auto" --max_batch_size=512 --output_path="results/concat-noop-64-768-1024-greedy" --tasks $TASKS

# BYTES OUT
lm-eval --model MoT --model_args "name=snimu/MoT_pad-lr_pull-yy_bpt-16_how-concat-copy_nlo-1_BTMdim-48-256-1024_niter-100000_773322,n=1" --num_fewshot 3 --batch_size="auto" --max_batch_size=512 --output_path="results/concat-copy-48-256-1024-N1-greedy" --tasks $TASKS
lm-eval --model MoT --model_args "snimu/MoT_pad-lr_pull-yy_bpt-16_how-concat-split_nlo-0_BTMdim-48-256-1024_niter-100000_773322,n=1" --num_fewshot 3 --batch_size="auto" --max_batch_size=512 --output_path="results/concat-split-48-256-1024-N1-greedy" --tasks $TASKS

lm-eval --model MoT --model_args "name=snimu/MoT_pad-lr_pull-yy_bpt-16_how-concat-copy_nlo-1_BTMdim-48-256-1024_niter-100000_773322,n=7" --num_fewshot 3 --batch_size="auto" --max_batch_size=512 --output_path="results/noop-copy-48-256-1024-N7-greedy" --tasks $TASKS
lm-eval --model MoT --model_args "snimu/MoT_pad-lr_pull-yy_bpt-16_how-concat-split_nlo-0_BTMdim-48-256-1024_niter-100000_773322,n=7" --num_fewshot 3 --batch_size="auto" --max_batch_size=512 --output_path="results/concat-split-48-256-1024-N7-greedy" --tasks $TASKS

lm-eval --model MoT --model_args "name=snimu/MoT_pad-lr_pull-yy_bpt-16_how-concat-copy_nlo-1_BTMdim-48-256-1024_niter-100000_773322,n=16" --num_fewshot 3 --batch_size="auto" --max_batch_size=512 --output_path="results/noop-copy-48-256-1024-N16-greedy" --tasks $TASKS
lm-eval --model MoT --model_args "snimu/MoT_pad-lr_pull-yy_bpt-16_how-concat-split_nlo-0_BTMdim-48-256-1024_niter-100000_773322,n=16" --num_fewshot 3 --batch_size="auto" --max_batch_size=512 --output_path="results/concat-split-48-256-1024-N16-greedy" --tasks $TASKS
