
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


def make_net_from_name(name: str) -> SpeedyLangNet:
    if "46M" in name:
        depth, width =  8, 384
    elif "240M" in name:
        depth, width = 21, 1024
    elif "773M" in name:
        depth, width = 35, 1664
    elif "1300M" in name:
        depth, width = 43, 2048
    else:
        raise ValueError(f"Unknown pretrained model {name}")
    
    num_heads = int(name.split("-")[-2].split("head")[0])
    
    return make_net({
        'depth': depth,
        'width': width,
        'linear_value': False,
        'num_heads': num_heads,
    })


def download_model(pretrained: str, cache_dir: str = ".") -> str:
    hf_hub_download(repo_id=pretrained, filename="model.safetensors", cache_dir=cache_dir)
    # Find model.safetensors in cache_dir (is in some subfolder)
    model_path = Path(cache_dir) / f"models--snimu--{pretrained.split('/')[1]}"
    model_path = list(model_path.glob("**/model.safetensors"))
    assert len(model_path) == 1, f"Expected exactly one model.safetensors file in cache_dir, got {model_path}"
    model_path = model_path[0]
    return str(model_path)


@torch.no_grad()
def generate(
        net, encoder, query: str, max_gen_tokens: int = 128, until: list[str] | None = None,
        choose_nth_best: int = 1,
) -> tuple[str, torch.Tensor, torch.Tensor]:
    # Encode the input tokens
    input_ids = encoder.encode_ordinary(query)
    input_ids = torch.tensor(input_ids, device="cuda", dtype=torch.int).unsqueeze(0)
    input_len = input_ids.shape[1]
    
    # Generate the output tokens
    output_str = []
    all_ids = input_ids
    for _ in range(max_gen_tokens):
        logits: torch.Tensor = net(all_ids)
        output_id = logits[:, -1, :50304].topk(choose_nth_best, dim=-1).indices[:, -1].item()  # ignore last token position, only decode valid token indices ( up to50304)
        char = encoder.decode([output_id])
        output_str.append(char)
        all_ids = torch.cat([all_ids, torch.tensor([output_id], device="cuda", dtype=torch.int).unsqueeze(0)], dim=1)
        if until and char in until:
            break

    # Get the logprops
    logprobs = F.softmax(net(all_ids), dim=-1).log()
    
    # Get the output text
    output_text = "".join(output_str)
    return output_text, logprobs, logprobs.squeeze()[input_len:]


@torch.no_grad()
def generate_with_mask(
    net: SpeedyLangNet, 
    encoder: tiktoken.Encoding, 
    query: str, 
    max_gen_tokens: int = 128, 
    mask: int = 50308,
    choose_nth_best: int = 1,
) -> tuple[str, torch.Tensor, torch.Tensor]:
    # Encode the input tokens
    input_ids = encoder.encode_ordinary(query)
    input_ids = torch.tensor(input_ids, device="cuda", dtype=torch.int).unsqueeze(0)
    input_len = input_ids.shape[1]

    all_ids = torch.cat(
        [
            input_ids, 
            torch.empty(
                (1, max_gen_tokens), 
                device="cuda", 
                dtype=torch.int,
            ).fill_(mask),
        ], 
        dim=1,
    )
    logits: torch.Tensor = net(all_ids)
    logprobs = F.log_softmax(logits, dim=-1)
    outputs = logits[:, input_len:, :50304].topk(choose_nth_best, dim=-1).indices[:, :, -1]
    outputs = outputs.squeeze().tolist()
    output_text = encoder.decode(outputs)
    
    return output_text, logprobs, logprobs.squeeze()[input_len:]


class CausalUl2(LM):
    def __init__(
            self,
            pretrained: str,
            **kwargs,
    ) -> None:
        super().__init__()
        self.net = make_net_from_name(pretrained)
        self.model_path = download_model(pretrained)
        safetensors.torch.load_model(self.net, self.model_path, device="cuda")

        self.encoder = tiktoken.get_encoding("gpt2")

    def loglikelihood(
            self, requests: list[Instance], disable_tqdm: bool = False
    ) -> list[tuple[float, bool]]:
        results = []
        for request in tqdm(requests, disable=disable_tqdm):
            # Get the input tokens
            query = request.args[0]
            target = request.args[1]
            target_ids = self.encoder.encode_ordinary(target)
            text, _, logprobs = generate(self.net, self.encoder, query, max_gen_tokens=len(target_ids))
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
            _, full_logprobs, _ = generate(self.net, self.encoder, query, max_gen_tokens=len(target_ids))
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

            text, _, _ = generate(self.net, self.encoder, query, max_gen_tokens, until)
            continuations.append(text)

        return continuations
    

try:
    register_model("causal-ul2")(CausalUl2)
except AssertionError:
    pass


def _test_model_loading():
    from rich import print

    """Test if the model weights are correctly loaded"""
    model_name_c = "snimu/causal-ul2-C-fineweb10BT-773M-26heads-lr090"
    model_name_r = "snimu/causal-ul2-R-fineweb10BT-773M-26heads-lr090"

    net_rand = make_net_from_name(model_name_c).to("cpu")
    net_c = make_net_from_name(model_name_c).to("cpu")
    for p1, p2 in zip(net_rand.parameters(), net_c.parameters()):
        p2.data.copy_(p1.data)
    assert all([torch.all(p1 == p2) for p1, p2 in zip(net_rand.parameters(), net_c.parameters())])

    model_path = download_model(model_name_c)
    safetensors.torch.load_model(net_c, model_path, device="cpu")
    assert not all([torch.all(p1 == p2) for p1, p2 in zip(net_rand.parameters(), net_c.parameters())])

    net_r = make_net_from_name(model_name_c).to("cpu")
    model_path = download_model(model_name_r)
    safetensors.torch.load_model(net_r, model_path, device="cpu")

    sentences = [
        "The cinnamon quail-thrush (Cinclosoma cinnamomeum) is a species of bird in the family Cinclosomatidae. Endemic to Australia, it is typically found in arid and semi-arid regions of the central part of the continent, spanning southwest Queensland, northwest New South Wales, northeastern South Australia, and the southeast of the Northern Territory. It is most commonly found among dry stony areas, especially",
        'Carcharodontosauridae (carcharodontosaurids; from the Greek carcharodontosauros: "shark-toothed lizards") is a group of carnivorous theropod dinosaurs. In 1931, Ernst Stromer named Carcharodontosauridae as a family, which, in modern paleontology, indicates a clade within Carnosauria. Carcharodontosaurids include some of the largest land predators ever known: Giganotosaurus, Mapusaurus, Carcharodontosaurus, and Tyrannotitan all rivaled Tyrannosaurus in size. Estimates give a maximum weight of',
        "The 2000–01 World Sevens Series was the second edition of the global circuit for men's national rugby sevens teams, organised by the International Rugby Board. The season ran from November 2000 to June 2001 and consisted of nine tournaments (originally 10 were scheduled, but one was cancelled).\n\nThe series was won by New Zealand, who won six of the nine tournaments. Australia won the other three tournaments, and",
        'Peregian Beach within the Sunshine Coast Region comprises continual residential development along the eastern coastal strip of sandy beaches. The David Low Way passes north to south through this area. Development to the west is constrained by',
        "Linton was the eldest son of Jabez Linton of Hardrigg Lodge, Dumfriesshire, by Jane, daughter of William Crocket of Grahamshill in the same county. He was born in 1801 at Kirkpatrick Fleming. He was educated at Edinburgh University, and graduated L.R.C.S. in 1826. But he had already utilised four summer vacations as surgeon on a whaler in the arctic regions. He entered the army medical department in 1826, graduated M.D. at Glasgow in 1834, and became staff surgeon of the first class in 1848. After serving in Canada, the Mediterranean, and the West Indies, he was appointed deputy inspector-general of hospitals of the first division of the army in the Crimea, was present in every action up to the battle of Balaclava, and had care of the barrack hospital in Scutari shortly after its establishment in 1854 until the British forces",
        "Two years later, Mozambique qualified for their third Africa Cup of Nations held in Burkina Faso. They were again placed in group D along with Morocco, Egypt and Zambia. Mozambique lost their first game against eventual tournament winners Egypt 2–0, both goals coming from Hossam Hassan. In their second game they again lost to Morocco 3–0, therefore eliminating them from",
        "Kirkman intended Freedom Ring to be an example of a superhero who demonstrated inexperience with his superpowers, as he felt that most superheroes quickly adjusting to their powers and having a successful superhero career did not reflect reality. When asked by a fan about the number of visibly gay comic book superheroes, Editor-in-Chief of Marvel Comics, Joe Quesada, also touted"
        'Portage-du-Fort is named after the portage trail which started here and would lead upstream around a set of falls on the Ottawa River.\n\nHowever, there are several hypotheses to explain the "Fort" portion. Among the most popular is the assumption that a fort was present here on the shore of the Ottawa River to keep provisions at the portage. It has been claimed that a fort called Dufort was flooded in the rapids at this location. However, some researchers argue that the fort in question has never existed and may be a reference to another fort at the mouth of the Coulonge River (after which modern Fort-Coulonge is named). Moreover, the word formerly did not always convey a military connotation and could be more or less synonymous with a village or hamlet, or even a post or warehouse which was fortified.[1]\n\nOne theory suggests that the name goes back to a custom of the Algonquins who would paint their bodies here and it was originally named Portage du Fard (French for "make-up"), which changed into "Fort".[1]\n\nAnother possibility is that Fort (French also for "strong") makes reference to the strength needed to haul the heavy canoes and supplies"',
        "The Country Club of Birmingham, previously known as Birmingham Country Club, located in Birmingham, Alabama, United States, was founded in 1898. It moved in 1900 from North Birmingham to Lakeview, then again in 1926 to a site in Shades Valley, now within the city of Mountain Brook. The Lakeview club hosted former president Theodore Roosevelt and several Women's Southern Golf Association tournaments.",
        "Saint Barthélemy was for many years a French commune forming part of Guadeloupe, which is an overseas region and department of France. In 2003 the island voted in favour of secession from Guadeloupe to form a separate overseas collectivity (collectivité d'outre-mer, abbreviated to COM) of France. The collectivity is one of four territories among the Leeward Islands in the northeastern Caribbean that make up the French West Indies, along with Saint Martin, Guadeloupe (200 kilometres (120 mi) southeast) and",
    ]

    net_c = net_c.to("cuda")
    net_r = net_r.to("cuda")

    encoder = tiktoken.get_encoding("gpt2")
    for sentence in sentences:
        completion_c1, _, _ = generate(net_c, encoder, sentence, max_gen_tokens=50)
        completion_r1, _, _ = generate(net_r, encoder, sentence, max_gen_tokens=50)
        completion_c2, _, _ = generate(net_c, encoder, sentence, max_gen_tokens=50, choose_nth_best=2)
        completion_r2, _, _ = generate(net_r, encoder, sentence, max_gen_tokens=50, choose_nth_best=2)
        completion_c3, _, _ = generate(net_c, encoder, sentence, max_gen_tokens=50, choose_nth_best=3)
        completion_r3, _, _ = generate(net_r, encoder, sentence, max_gen_tokens=50, choose_nth_best=3)
        # mask_completion_c1, _, _ = generate_with_mask(net_c, encoder, sentence, max_gen_tokens=50)
        # mask_completion_r1, _, _ = generate_with_mask(net_r, encoder, sentence, max_gen_tokens=50)
        # mask_completion_c2, _, _ = generate_with_mask(net_c, encoder, sentence, max_gen_tokens=50, choose_nth_best=2)
        # mask_completion_r2, _, _ = generate_with_mask(net_r, encoder, sentence, max_gen_tokens=50, choose_nth_best=2)
        # mask_completion_c3, _, _ = generate_with_mask(net_c, encoder, sentence, max_gen_tokens=50, choose_nth_best=3)
        # mask_completion_r3, _, _ = generate_with_mask(net_r, encoder, sentence, max_gen_tokens=50, choose_nth_best=3)
        print(
            f"\n\n{sentence=}\n\n"
            f"{completion_c1=}\n"
            f"{completion_r1=}\n\n"
            f"{completion_c2=}\n"
            f"{completion_r2=}\n\n"
            f"{completion_c3=}\n"
            f"{completion_r3=}\n\n"
            # f"{mask_completion_c1=}\n"
            # f"{mask_completion_r1=}\n\n"
            # f"{mask_completion_c2=}\n"
            # f"{mask_completion_r2=}\n\n"
            # f"{mask_completion_c3=}\n"
            # f"{mask_completion_r3=}\n\n"
        ) 


if __name__ == "__main__":
    _test_model_loading()
