
import argparse
from time import perf_counter

from lm_eval.api.instance import Instance
from lm_eval.models.mot import MoTModel


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-gen-toks", type=int, default=1024)
    parser.add_argument("--to-file", type=str, default=None)
    return parser.parse_args()


def main():
    args = get_args()
    model = MoTModel(args.name, temperature=args.temperature)
    while True:
        query = input("\n\n New, independent Query: ")
        if not query:
            break
        print("\n\n")
        instance = Instance(
            "generate_until",
            doc=dict(),
            arguments=(query, dict(max_gen_toks=args.max_gen_toks)),
            idx=0,
            metadata=(None, None, None),
            resps=[],
            filtered_resps={},
        )
        t0 = perf_counter()
        response = model.generate_until([instance]).pop()
        t1 = perf_counter()
        print(response)
        print(f"\n\nIn {(t1-t0):.2f} seconds")
        if args.to_file:
            with open(args.to_file, "a") as f:
                f.write(f"\n\nQuery: {query}\n\nResponse: {response}\n\n")


if __name__ == "__main__":
    main()
