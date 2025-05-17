
import argparse

from lm_eval.api.instance import Instance
from lm_eval.models.mot import MoT


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-gen-toks", type=int, default=1024)
    return parser.parse_args()


def main():
    args = get_args()
    model = MoT(args.name, temperature=args.temperature)
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
        responses = model.generate_until(instance)
        for response in responses:
            print(response)


if __name__ == "__main__":
    main()
