
import argparse
import json
import os
from time import perf_counter
from typing import Generator, Literal

import numpy as np
import polars as pl
import tiktoken
import torch
from datasets import load_dataset
from tqdm import tqdm

from lm_eval.api.instance import Instance
from lm_eval.models.mot import MoTModel


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", nargs="+")
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-gen-toks", type=int, default=600)
    parser.add_argument("--to-file", type=str, default=None)
    parser.add_argument("--queries-file", type=str, default="queries.json")
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--tokens-in", type=int, nargs="+", default=None)
    parser.add_argument("--n", type=int, default=1, help="only relevant when outputs are bytes")
    parser.add_argument("--dataset", nargs="+", default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-tokens", type=int, default=1024)

    args = parser.parse_args()
    args.name = [args.name] if isinstance(args.name, str) else args.name
    if args.tokens_in is None:
        args.tokens_in = [20, 100, 500]
    if args.dataset is None:
        args.dataset = ["wikipedia"]
    return args


def make_instance(query: str, max_gen_toks: int) -> Instance:
    return Instance(
        "generate_until",
        doc=dict(),
        arguments=(query, dict(max_gen_toks=max_gen_toks)),
        idx=0,
        metadata=(None, None, None),
        resps=[],
        filtered_resps={},
    )


def generate(
        model: MoTModel,
        queries: list[str],
        enc: tiktoken.Encoding,
        num_samples: int,
        tokens_in: list[int],
        max_gen_toks: int,
        to_file: str,
):
    loop = tqdm(queries)
    for query in loop:
        results = [{"name": query["name"], "tokens_in": [], "responses": [], "times": []}]
        query_tokens = enc.encode(query["text"])
        for _ in range(num_samples):
            for toks_in in tokens_in:
                text = enc.decode(query_tokens[:toks_in])
                t0 = perf_counter()
                response = model.generate_until([make_instance(text, max_gen_toks)]).pop()
                t1 = perf_counter()
                results[-1]["responses"].append(response)
                results[-1]["times"].append(t1-t0)
                results[-1]["tokens_in"].append(toks_in)
        if to_file:
            if os.path.exists(to_file):
                with open(to_file, "r") as f:
                    results = json.loads(f.read()) + results
            with open(to_file, "w") as f:
                f.write(json.dumps(results, indent=2))


def dataset_generator(
        enc: tiktoken.Encoding,
        ds_name: Literal["wikipedia"] = "wikipedia",
        batch_size: int = 64,
        num_tokens: int = 1024,
) -> Generator[None, None, torch.Tensor]:
    if ds_name == "wikipedia":
        ds = load_dataset("wikimedia/wikipedia", "20231101.en", streaming=True)
        buffer = []
        tokens = []
        for item in ds.iter_rows():
            text = item["text"]
            buffer.append(text)
            if len(buffer) >= batch_size - len(tokens):
                tokens = tokens + enc.encode_batch(buffer, disallowed_special=[])
                tokens = [toks[:num_tokens] for toks in tokens if len(toks) >= num_tokens]
                buffer = []
            if len(tokens) >= batch_size:
                yield torch.tensor(tokens[:batch_size], device="cuda", requires_grad=False)
                tokens = []
    else:
        raise NotImplementedError(f"Unknown dataset {ds_name}")


def loss_pplx(
        model_names: list[str],
        dataset_names: list[str],
        enc: tiktoken.Encoding,
        num_tokens: int = 1024,
        batch_size: int = 64,
        to_file_csv: str = "losses_pplxs.csv",
):
    results = {"model": [], "dataset": [], "loss": [], "perplexity": []}
    for model_name in model_names:
        print(model_name)
        model = MoTModel(model_name)
        loop = tqdm(dataset_names)
        for ds_name in loop:
            dataloader = dataset_generator(enc, ds_name, batch_size, num_tokens)
            losses = []
            for batch_num, batch in enumerate(dataloader):
                loop.set_description(f"{model_name} on {ds_name}; batch {batch_num}")
                loss, pplx = model.batch_loss_pplx(batch)
                losses.append(loss)

            loss = np.mean(losses)
            pplx = np.exp(loss)
            results["model"] = model_name
            results["dataset"] = ds_name
            results["loss"] = loss
            results["pplx"] = pplx
            print(f"{model_name} on {ds_name}: {loss=:.2f}, {pplx=:.2f}")
    df = pl.DataFrame(results)
    df.write_csv(to_file_csv)


def main():
    args = get_args()
    queries = None
    enc = tiktoken.encoding_for_model("gpt-2")
    with open(args.queries_file, "r") as f:
        queries = json.loads(f.read())
    if args.generate:
        model = MoTModel(args.name.pop(), temperature=args.temperature, n=args.n)
        print("Warming up...")
        model.generate_until([make_instance("Hello, my name is", 10)])
        generate(
            model=model,
            queries=queries,
            enc=enc,
            num_samples=args.num_samples,
            tokens_in=args.tokens_in,
            max_gen_toks=args.max_gen_toks,
            to_file=args.to_file,
        )
    else:
        loss_pplx(
            model_names=args.name,
            dataset_names=args.dataset,
            enc=enc,
            num_tokens=args.num_tokens,
            batch_size=args.batch_size,
            to_file_csv=args.to_file,
        )


if __name__ == "__main__":
    main()
