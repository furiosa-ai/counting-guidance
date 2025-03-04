import argparse
import random
import inflect
from pathlib import Path

import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--obj", default="data/obj_v1.txt")
    parser.add_argument("--out_file", default="data/prompts_v2.yaml")
    parser.add_argument("--range", type=int, nargs=2, default=(1, 20))
    parser.add_argument("--examples_per_obj", type=int, default=1)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--max_objs_per_example", type=int, default=1)
    args = parser.parse_args()


    in_file = Path(args.obj)

    with open(in_file, "r") as f:
        objs = [line.rstrip() for line in f.readlines()]

    min_count, max_count = args.range
    random.seed(args.seed)

    data = []
    p = inflect.engine()

    idx = 0
    for obj_singular in objs:
        for _ in range(args.examples_per_obj):
            for count in range(min_count, max_count + 1):
                assert args.max_objs_per_example == 1

                obj = obj_singular
                if obj not in ("glasses", ):
                    obj = p.plural_noun(obj) if count > 1 else obj
                prompt = p.number_to_words(count) + " " + obj

                data.append({
                    "prompt_idx": idx,
                    "prompt": prompt,
                    "objs": [obj],
                    "counts": [count],
                    "objs_singular": [obj_singular]
                })

                idx += 1

    with open(args.out_file, "w") as f:
        yaml.dump(data, f)

 
if __name__ == "__main__":
    main()
