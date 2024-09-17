
from typing import Any, Literal
from pathlib import Path

from tqdm import tqdm
import torch
import safetensors.torch
import torch.nn as nn
import torch.nn.functional as F
import einops
import tiktoken
from huggingface_hub import hf_hub_download

from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from lm_eval.api.instance import Instance


max_seq_len = 4096


with torch.no_grad():
    # Create the base arrays for the learnable linear positional bias. This helps save some memory consumption & processing time
    bias_range                    = torch.arange(-max_seq_len+1, 1).to("cuda", dtype=torch.bfloat16)
    position_bias_base            = bias_range.unsqueeze(0) - bias_range.unsqueeze(1)
    negative_infinity_matrix_base = torch.empty_like(position_bias_base).fill_(-float("inf"))
    causal_mask = torch.tril(torch.ones((max_seq_len, max_seq_len), device="cuda", dtype=torch.bool))


class LatentAttentionBlock(nn.Module):
    """ Efficient fused latent-space attention block. Linear keys and queries, nonlinear values."""
    def __init__(self, width: int,  depth: int, linear_value: bool, num_heads: int):
        super().__init__()
        # Layer dim parameters. Play around with these, there's likely some undiscovered stuff still!
        self.dim        = width
        self.qk_dim     = self.dim//8
        self.v_dim      = width
        self.expand_dim = width * 2
        self.linear_value = linear_value 
        self.num_heads = num_heads

        # Main layer weights
        self.norm    = nn.LayerNorm(self.dim, bias=False)
        self.expand  = nn.Parameter(.5 * 1./width**.5 * 1./2                              * torch.randn(2*self.qk_dim+2*self.expand_dim, self.dim))
        self.project = nn.Parameter(1. * 1./width**.5 * 1./2 * 1./depth * torch.randn((self.dim, self.expand_dim)))

        # Learnable linear positional encodings. Similar to but different than https://arxiv.org/abs/2108.12409
        # Has a high lr mult applied to it so that each layer can learn its own attention scale.
        self.position_bias_mult = nn.Parameter(torch.tensor(1., device='cuda'))

    def forward(self, x, attn_mask: torch.Tensor):
        residual = x

        # Make additive attention mask, scaled by a learned mult for the position bias (lets us learn dynamic attention ranges per layer as needed)
        attn_mask_with_positional_bias = torch.where(attn_mask, F.softplus(self.position_bias_mult) * position_bias_base[:x.shape[1], :x.shape[1]], negative_infinity_matrix_base[:x.shape[1], :x.shape[1]])
        
        # Shared LayerNorm for linear layers and attention
        x = self.norm(x)

        # Fused into one kernel for memory+speed/etc
        query, key, linear, pre_gelu = F.linear(x, self.expand).split((self.qk_dim, self.qk_dim, self.expand_dim, self.expand_dim), dim=-1)

        # Compute GeGLU (one portion of the channels this will stay locally, another will become the nonlinear value for attention)
        geglu = linear * F.gelu(pre_gelu)

        # Partition between the input values and the v dim values
        if self.linear_value:
            geglu_local, _ = geglu.split((self.expand_dim-self.v_dim, self.v_dim), -1)
            _, geglu_attention_value = pre_gelu.split((self.expand_dim-self.v_dim, self.v_dim), -1)
        else:
            geglu_local, geglu_attention_value = geglu.split((self.expand_dim-self.v_dim, self.v_dim), -1)

        if self.num_heads > 1:
            if len(attn_mask_with_positional_bias.shape) == 3:
                attn_mask_with_positional_bias = einops.repeat(attn_mask_with_positional_bias, 'b s1 s2 -> b h s1 s2', h=self.num_heads)
            query, key, geglu_local, geglu_attention_value = map(lambda x: einops.rearrange(x, 'b n (h d) -> b h n d', h=self.num_heads), (query, key, geglu_local, geglu_attention_value))

        # Compute attention. Something to note is that there are no attention heads here. This seemed to work a bit better, maybe due to not needing memory `.contiguous()` calls or similar
        attention = F.scaled_dot_product_attention(query, key, geglu_attention_value, attn_mask=attn_mask_with_positional_bias)

        if self.num_heads > 1:
            attention = einops.rearrange(attention, 'b h n d -> b n (h d)')
            geglu_local = einops.rearrange(geglu_local, 'b h n d -> b n (h d)')

        # Output linear layer
        out = F.linear(torch.cat([geglu_local, attention], dim=-1), self.project)

        # Add to residual
        x = residual + out

        return x


#############################################
#            Network Definition             #
#############################################

# This may seem like an odd way to define a network, but it's a bit easier to hack into/make quick changes than other methods
class SpeedyLangNet(nn.Module):
    def __init__(self, network_dict):
        super().__init__()
        self.net_dict = network_dict

    def forward(self, x, mode: Literal['causal', 'noncausal', 'mixed'] = "causal"):
        # Look up the input embeddings from the input tokens
        attn_mask = causal_mask[:x.shape[1], :x.shape[1]]
        x = self.net_dict['embedding'](x)
        for attn_block in self.net_dict['attn_layers']:
            x = attn_block(x, attn_mask=attn_mask) # note: residuals are included in the block definitions for these layers
        x = self.net_dict['norm'](x)
        x = self.net_dict['outputs'](x)
        return x
    

def make_attn(settings: dict[str, Any]):
    # You can parametrically change anything you want about the attn blocks here
    return LatentAttentionBlock(
        settings['width'], settings['depth'], settings['linear_value'], settings['num_heads']
    )


def make_net(settings: dict[str, Any]):
    total_num_tokens = 50310
    network_dict = nn.ModuleDict({
        'embedding': nn.Embedding(total_num_tokens, settings['width'], scale_grad_by_freq=True),
        'attn_layers': nn.ModuleList([make_attn(settings) for _ in range(settings['depth'])]),
        'norm': nn.LayerNorm(settings['width'], bias=False),
        'outputs': nn.Linear(settings['width'], total_num_tokens, bias=False),
    })
    net = SpeedyLangNet(network_dict)
    net = net.to("cuda", dtype=torch.bfloat16)
    net.eval()

    # Initialize the embedding and output matrixes, with weights scaled based upon the dimensionality of the network.
    torch.nn.init.normal_(net.net_dict['embedding'].weight.data, std=.25*1./settings['width']**.5)
    torch.nn.init.normal_(net.net_dict['outputs']  .weight.data, std=.5 *1./settings['width']**.5)

    return net


def download_model(pretrained: str, cache_dir: str = ".") -> str:
    hf_hub_download(repo_id=pretrained, filename="model.safetensors", cache_dir=cache_dir)
    # Find model.safetensors in cache_dir (is in some subfolder)
    model_path = Path(cache_dir) / f"models--snimu--{pretrained.split('/')[1]}"
    model_path = list(model_path.glob("**/model.safetensors"))
    assert len(model_path) == 1, f"Expected exactly one model.safetensors file in cache_dir, got {model_path}"
    model_path = model_path[0]
    return str(model_path)


@register_model("causal-ul2")
class CausalUl2(LM):
    def __init__(
            self,
            pretrained: str,
            **kwargs,
    ) -> None:
        super().__init__()
        if "46M" in pretrained:
            depth, width =  8, 384
        elif "240M" in pretrained:
            depth, width = 21, 1024
        elif "773M" in pretrained:
            depth, width = 35, 1664
        elif "1300M" in pretrained:
            depth, width = 43, 2048
        else:
            raise ValueError(f"Unknown pretrained model {pretrained}")
        
        num_heads = int(pretrained.split("-")[-2].split("head")[0])
        
        self.net = make_net({
            'depth': depth,
            'width': width,
            'linear_value': False,
            'num_heads': num_heads,
        })
        self.model_path = download_model(pretrained)
        safetensors.torch.load_model(self.net, self.model_path, device="cuda")

        self.encoder = tiktoken.get_encoding("gpt2")

    @torch.no_grad()
    def generate(
            self, query: str, max_gen_tokens: int = 128, until: list[str] | None = None
    ) -> tuple[str, torch.Tensor, torch.Tensor]:
        # Encode the input tokens
        input_ids = self.encoder.encode_ordinary(query)
        input_ids = torch.tensor(input_ids, device="cuda", dtype=torch.int).unsqueeze(0)
        input_len = input_ids.shape[1]
        
        # Generate the output tokens
        output_str = []
        all_ids = input_ids
        for _ in range(max_gen_tokens):
            output_id = self.net(input_ids)[:, -1].argmax(-1).item()
            char = self.encoder.decode([output_id])
            output_str.append(char)
            all_ids = torch.cat([all_ids, torch.tensor([output_id], device="cuda", dtype=torch.int).unsqueeze(0)], dim=1)
            if until and char in until:
                break

        # Get the logprops
        logprobs = F.softmax(self.net(all_ids), dim=-1).log()
        
        # Get the output text
        output_text = "".join(output_str)
        return output_text, logprobs, logprobs.squeeze()[input_len:]

    def loglikelihood(
            self, requests: list[Instance], disable_tqdm: bool = False
    ) -> list[tuple[float, bool]]:
        results = []
        for request in tqdm(requests, disable=disable_tqdm):
            # Get the input tokens
            query = request.args[0]
            target = request.args[1]
            target_ids = self.encoder.encode_ordinary(target)
            text, _, logprobs = self.generate(query, max_gen_tokens=len(target_ids))
            reduced_logprobs = [logprobs[i, t].item() for i, t in enumerate(target_ids)]
            ll = sum(reduced_logprobs)
            is_greedy = text == target
            results.append((ll, is_greedy))

        return results


    def loglikelihood_rolling(
            self, requests: list[Instance], disable_tqdm: bool = False
    ) -> list[float]:
        lls = []
        for request in tqdm(requests, disable=disable_tqdm):
            # Get the input tokens
            query = request.args[0]
            target = request.args[1]
            target_ids = self.encoder.encode_ordinary(target)
            _, full_logprobs, _ = self.generate(query, max_gen_tokens=len(target_ids))
            reduced_logprobs = [full_logprobs[i, t] for i, t in enumerate(target_ids)]
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

            text, _, _ = self.generate(query, max_gen_tokens, until)
            continuations.append(text)

        return continuations