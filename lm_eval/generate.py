
import argparse
import json
import os
from time import perf_counter

from datasets import load_dataset

from lm_eval.api.instance import Instance
from lm_eval.models.mot import MoTModel


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-gen-toks", type=int, default=1024)
    parser.add_argument("--to-file", type=str, default=None)
    parser.add_argument("--queries-file", type=str, default=None)
    parser.add_argument("--num-samples", type=int, default=1)
    return parser.parse_args()


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


def make_queries_file(filename: str):
    if os.path.exists(filename):
        return
    # Download WikiText-103 from HF, pick out the first 100 queries
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1")
    queries = ds["train"].select(range(100))
    with open(filename, "w") as f:
        f.write(json.dumps(queries, indent=2))


def main():
    args = get_args()
    model = MoTModel(args.name, temperature=args.temperature)
    print("Warming up...")
    model.generate_until([make_instance("Hello, my name is", 10)])
    queries = None
    if args.queries_file:
        make_queries_file(args.queries_file)
        with open(args.queries_file, "r") as f:
            queries = iter(json.loads(f.read()))
    while True:
        if queries:
            try:
                query = next(queries)
                print(f"\n\nNew Query: {query}")
            except StopIteration:
                break
        else:
            query = input("\n\n New, independent Query: ")
        if not query:
            break
        print("\n\n")
        results = [{"query": query, "responses": [], "times": []}]
        for _ in range(args.num_samples):
            t0 = perf_counter()
            response = model.generate_until([make_instance(query, args.max_gen_toks)]).pop()
            t1 = perf_counter()
            results[-1]["responses"].append(response)
            results[-1]["times"].append(t1-t0)
        if args.to_file:
            if os.path.exists(args.to_file):
                with open(args.to_file, "r") as f:
                    results = json.loads(f.read()) + results
            with open(args.to_file, "w") as f:
                f.write(json.dumps(results, indent=2))
        for response in results[-1]["responses"]:
            print("\n\n" + response)
        print(f"\n\nIn {sum(results[-1]['times']) / len(results[-1]['times']):.2f} seconds per generation.")


if __name__ == "__main__":
    main()
