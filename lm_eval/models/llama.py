
"""
INSTRUCTIONS:

- you need to run `huggingface-cli login` once and add your token
- you need access to the meta-llama weights

ON LAMBDA LABS:

- you need to run `pip install -U torch torchvision` before running this script
"""

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model


@torch.no_grad()
def generate(
        model, tokenizer, query: str, max_gen_tokens: int = 128, until: list[str] | None = None,
        choose_nth_best: int = 1, merge_tokens: bool = False,
) -> str:
    # Encode the input tokens
    input_ids = tokenizer(query, return_tensors="pt").input_ids.to(model.device)
    
    # Generate the output tokens
    output_str = []
    all_ids = input_ids
    
    for _ in range(max_gen_tokens):
        # If merge_tokens is True, convert current sequence to text and back to tokens
        if merge_tokens and len(output_str) > 0:
            current_text = query + "".join(output_str)
            all_ids = tokenizer(current_text, return_tensors="pt").input_ids.to(model.device)
        
        outputs = model(all_ids)
        logits = outputs.logits
        output_id = logits[:, -1].topk(choose_nth_best, dim=-1).indices[:, -1].item()
        
        # Decode single token
        char = tokenizer.decode([output_id])
        output_str.append(char)
        
        if not merge_tokens:
            all_ids = torch.cat([all_ids, torch.tensor([[output_id]], device=model.device)], dim=1)
        
        if until and char in until:
            break

    output_text = "".join(output_str)
    return output_text


@torch.no_grad()
def get_logprobs(
        model, tokenizer, query: str, target: str | None = None, merge_tokens: bool = False
) -> torch.Tensor:
    # Encode the input tokens
    input_encoding = tokenizer(query, return_tensors="pt")
    input_ids = input_encoding.input_ids.to(model.device)
    input_len = input_ids.shape[1]
    
    if target is not None:
        target_encoding = tokenizer(target, return_tensors="pt")
        target_ids = target_encoding.input_ids.to(model.device)
        
        if merge_tokens:
            # Merge by converting to text and back to tokens
            full_text = query + tokenizer.decode(target_ids[0, :-1])
            all_ids = tokenizer(full_text, return_tensors="pt").input_ids.to(model.device)
        else:
            # Original behavior
            all_ids = torch.cat([input_ids, target_ids[:, :-1]], dim=1)
    else:
        all_ids = input_ids
    
    # Get logits and convert to log probabilities
    outputs = model(all_ids)
    logits = outputs.logits
    logprobs = F.log_softmax(logits[:, input_len-1:], dim=-1)
    
    return logprobs


class LlamaModel(LM):
    def __init__(
            self,
            instruct: bool = False,
            merge_tokens: bool = False,
            **kwargs,
    ):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_name = "meta-llama/Llama-3.2-1B-instruct" if instruct else "meta-llama/Llama-3.2-1B"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(self.device)
        self.merge_tokens = merge_tokens

    def loglikelihood(
            self, requests: list[Instance], disable_tqdm: bool = False
    ) -> list[tuple[float, bool]]:
        results = []
        for request in tqdm(requests, disable=disable_tqdm):
            query, target = request.args
            target_ids = self.tokenizer(target, return_tensors="pt").input_ids[0]
            
            logprobs = get_logprobs(self.model, self.tokenizer, query, target, merge_tokens=self.merge_tokens)
            reduced_logprobs = [logprobs[0, i, tid].item() for i, tid in enumerate(target_ids)]
            ll = sum(reduced_logprobs)
            
            # For is_greedy, check if target tokens have highest probability
            is_greedy = all(tid == logprobs[0, i, :].argmax().item() for i, tid in enumerate(target_ids))
            
            results.append((ll, is_greedy))
        return results
    
    def loglikelihood_rolling(self, requests: list[Instance], disable_tqdm: bool = False) -> list[float]:
        lls = []
        for request in tqdm(requests, disable=disable_tqdm):
            query, target = request.args
            target_ids = self.tokenizer(target, return_tensors="pt").input_ids[0]
            
            logprobs = get_logprobs(self.model, self.tokenizer, query, target, merge_tokens=self.merge_tokens)
            reduced_logprobs = [logprobs[0, i, tid].item() for i, tid in enumerate(target_ids)]
            ll = sum(reduced_logprobs)
            lls.append(ll)

        return lls

    def generate_until(
            self, requests: list[Instance], disable_tqdm: bool = False
    ) -> list[str]:
        continuations = []
        for request in tqdm(requests, disable=disable_tqdm):
            # Get the input tokens
            query = request.args[0]
            until = request.args[1].get("until", ["</s>"])
            max_gen_tokens = request.args[1].get("max_gen_tokens", 128)

            text = generate(
                self.model, self.tokenizer, query, max_gen_tokens, 
                until, merge_tokens=self.merge_tokens,
            )
            continuations.append(text)

        return continuations
    

try:
    register_model("llama-merge")(LlamaModel)
except AssertionError:
    pass
