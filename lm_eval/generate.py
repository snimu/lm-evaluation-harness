
import argparse
import json
import os
from time import perf_counter

import tiktoken
from tqdm import tqdm

from lm_eval.api.instance import Instance
from lm_eval.models.mot import MoTModel


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-gen-toks", type=int, default=600)
    parser.add_argument("--to-file", type=str, default=None)
    parser.add_argument("--queries-file", type=str, default="queries.json")
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--tokens-in", type=int, nargs="+", default=None)

    args = parser.parse_args()
    if args.tokens_in is None:
        args.tokens_in = [20, 100, 500]
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


def main():
    args = get_args()
    model = MoTModel(args.name, temperature=args.temperature)
    print("Warming up...")
    model.generate_until([make_instance("Hello, my name is", 10)])
    queries = None
    enc = tiktoken.encoding_for_model("gpt-2")
    with open(args.queries_file, "r") as f:
        queries = json.loads(f.read())
    loop = tqdm(queries)
    for query in loop:
        results = [{"name": query["name"], "tokens_in": [], "responses": [], "times": []}]
        query_tokens = enc.encode(query["text"])
        for _ in range(args.num_samples):
            for tokens_in in args.tokens_in:
                text = enc.decode(query_tokens[:tokens_in])
                t0 = perf_counter()
                response = model.generate_until([make_instance(text, args.max_gen_toks)]).pop()
                t1 = perf_counter()
                results[-1]["responses"].append(response)
                results[-1]["times"].append(t1-t0)
                results[-1]["tokens_in"].append(tokens_in)
        if args.to_file:
            if os.path.exists(args.to_file):
                with open(args.to_file, "r") as f:
                    results = json.loads(f.read()) + results
            with open(args.to_file, "w") as f:
                f.write(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
