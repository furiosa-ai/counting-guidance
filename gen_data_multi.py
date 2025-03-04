import argparse
import random
import inflect
from pathlib import Path

import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--obj", default="data/obj_v1.txt")
    parser.add_argument("--out_file", default="data/prompts_multi_v1.yaml")
    parser.add_argument("--range", type=int, nargs=2, default=(2, 20))
    parser.add_argument("--examples_per_obj", type=int, default=2)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()


    in_file = Path(args.obj)

    with open(in_file, "r") as f:
        objs = [line.rstrip() for line in f.readlines()]

    min_count, max_count = args.range
    random.seed(args.seed)

    data = []
    p = inflect.engine()

    prompt_objs_singular = []
    for i, obj_singular in enumerate(objs):
        for j, obj_singular2 in enumerate(objs[i+1:]):
            prompt_objs_singular.append([obj_singular, obj_singular2])

    idx = 0
    for objs_singular in prompt_objs_singular:
        for _ in range(args.examples_per_obj):
            counts = [random.randint(min_count, max_count) for _ in range(len(objs_singular))]
            
            objs = []
            
            for i, obj in enumerate(objs_singular):
                if obj not in ("glasses", ):
                    obj = p.plural_noun(obj) if counts[i] > 1 else obj
                objs.append(obj)

            prompt = " and ".join([p.number_to_words(count) + " " + obj for count, obj in zip(counts, objs)])

            data.append({
                "prompt_idx": idx,
                "prompt": prompt,
                "objs": objs,
                "counts": counts,
                "objs_singular": list(objs_singular)
            })

            idx += 1

    with open(args.out_file, "w") as f:
        yaml.dump(data, f, Dumper=yaml.CSafeDumper)

 
if __name__ == "__main__":
    main()
