import argparse
from math import ceil
import torch.multiprocessing as mp
from typing import List, Dict
import torch
import cv2
from PIL import Image

from pathlib import Path

import os
import yaml
import pandas as pd

from models.counting_guidance_pipeline import CountingGuidancePipeline
from config import RunConfig
from utils.ptp_utils import AttentionStore
from utils import ptp_utils


def setup():
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    stable = CountingGuidancePipeline.from_pretrained("CompVis/stable-diffusion-v1-4").to(device)
    #stable = CountingGuidancePipeline.from_pretrained("CompVis/stable-diffusion-v1-5").to(device)
    # tokenizer = stable.tokenizer
    
    return stable


def run_on_prompt(prompt: List[str],
                  model: CountingGuidancePipeline,
                  controller: AttentionStore,
                  token_indices: List[int],
                  seed: torch.Generator,
                  config: RunConfig,
                  **kwargs) -> Image.Image:
    if controller is not None:
        ptp_utils.register_attention_control(model, controller)

    if "num_inference_steps" not in kwargs:
        kwargs["num_inference_steps"] = config.n_inference_steps

    outputs = model(prompt=prompt,
                    attention_store=controller,
                    indices_to_alter=token_indices,
                    attention_res=config.attention_res,
                    guidance_scale=config.guidance_scale,
                    generator=seed,
                    max_iter_to_alter=config.max_iter_to_alter,
                    run_standard_sd=config.run_standard_sd,
                    thresholds=config.thresholds,
                    scale_factor=config.scale_factor,
                    scale_range=config.scale_range,
                    smooth_attentions=config.smooth_attentions,
                    sigma=config.sigma,
                    kernel_size=config.kernel_size,
                    sd_2_1=config.sd_2_1,
                    # scale_count = config.scale_count,
                    **kwargs)
    image = outputs.images[0]
    return image


# configurable parameters (see RunConfig for all parameters)
# scale factor- intensity of shift by gradient
# thresholds- a dictionary for iterative refinement mapping the iteration number to the attention threshold
# max_iter_to_alter- maximal inference timestep to apply Attend-and-Excite
def run_and_display(stable,
                    prompts: List[str],
                    controller: AttentionStore,
                    indices_to_alter: List[int],
                    generator: torch.Generator,
                    run_standard_sd: bool = False,
                    scale_factor: int = 20,
                    thresholds: Dict[int, float] = {0: 0.05, 10: 0.5, 20: 0.8},
                    max_iter_to_alter: int = 200,#25
                    display_output: bool = False,
                    **kwargs):
    config = RunConfig(prompt=prompts[0],
                       run_standard_sd=run_standard_sd,
                       scale_factor=scale_factor,
                       thresholds=thresholds,
                       max_iter_to_alter=max_iter_to_alter)
    image = run_on_prompt(model=stable,
                          prompt=prompts,
                          controller=controller,
                          token_indices=indices_to_alter,
                          seed=generator,
                          config=config,
                          **kwargs)

    return image


def run_prompt(stable, seed, prompt, token_names, token_counts, *args, **kwargs):
    prompt_words = prompt.split(" ")
    token_indices = [prompt_words.index(t) + 1 for t in token_names]

    print(prompt)
    print(", ".join(f"{prompt_words[i - 1]} x{c}" for i, c in zip(token_indices, token_counts)))

    g = torch.Generator('cuda').manual_seed(seed)
    prompts = [prompt]
    controller = AttentionStore()
    image = run_and_display(stable, prompts=prompts,
                            controller=controller,
                            indices_to_alter=token_indices,
                            token_counts=token_counts,
                            generator=g,
                            run_standard_sd=True,
                            display_output=True,
                            *args, **kwargs
                            )
    
    return image


def demo_single(stable):
    prompt = "ten apples on a table"
    token_names = ["apples"]
    token_counts = [10]

    counting_loss_scales = [10]
    attention_loss_start = 25
    attention_loss_scale = 10
    grounding_dino_crit = None
    grounding_dino_scale = None  # 1e5

    seed = 0
    #seeds = [1,2]

    img = run_prompt(
        stable,
        seed,
        prompt,
        token_names,
        token_counts, 
        grounding_dino_crit=grounding_dino_crit,
        grounding_dino_scale=grounding_dino_scale,
        counting_loss_scales=counting_loss_scales,
        attention_loss_start=attention_loss_start,
        attention_loss_scale=attention_loss_scale,
        save_imgs=False,
    )

    img.save("out.png")


def run_all(stable, inputs, config, save_path, skip_exist=True):
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    with open(save_path / "cfg.yaml", "w") as f:
        yaml.safe_dump(config, f)

    if config.get("num_samples", None) is not None:
        inputs = inputs[:config.pop("num_samples")]

    for sample in inputs:
        if "cat" in sample:
            out_dir = save_path / sample["cat"] 
            out_dir.mkdir(exist_ok=True, parents=True)
        else:
            out_dir = save_path


        prompt = sample["prompt"].rstrip()
        if prompt[-1] == ".":
            prompt = prompt[:-1]

        # img_name = f"{sample['prompt_idx']}_{prompt.replace(' ', '_')}"
        img_name = prompt
        img_file_save_path = out_dir / (img_name + ".png")

        if skip_exist and img_file_save_path.is_file():
            continue

        # prompt_idx, prompt, token_names, token_counts = sample[]

        img = run_prompt(
            stable,
            prompt=sample["prompt"],
            token_names=sample["objs"],
            token_counts=sample["counts"], 
            save_imgs=False,
            **config
        )

        img.save(str(img_file_save_path))


def load_hrs_data(path):
    def find_all(s, q, start):
        s = s.rstrip()
        if s[-1] == ".":
            s = s[:-1]

        s = s.split(" ")[start:]
        return [i for i, w in enumerate(s) if w == q]

    df = pd.read_csv(path)

    data = []
    
    for prompt_idx, (n1, obj1, n2, obj2, synthetic_prompt) in enumerate(zip(*[df.get(k, None) for k in ["n1", "obj1", "n2", "obj2", "synthetic_prompt"]])):
        synthetic_prompt = synthetic_prompt.rstrip()
        if synthetic_prompt[-1] == ".":
            synthetic_prompt = synthetic_prompt[:-1]

        token_names = [obj1]
        token_counts = [n1]

        if n2 > 0:
            token_names += [obj2]
            token_counts += [n2]

        prompt_valid = True
        token_names_adj = []
        cur_idx = 0
        for token_name in token_names:
            idx = find_all(synthetic_prompt, token_name, cur_idx)

            if len(idx) == 0:
                token_name = token_name + "s"
                idx = find_all(synthetic_prompt, token_name, cur_idx)

            if len(idx) == 1:
                token_names_adj += [token_name]
                cur_idx = idx[0] + 1
            else:
                prompt_valid = False
                print("Invalid prompt:", synthetic_prompt)

        if prompt_valid:
            if len(token_names_adj) > 1:
                assert token_names_adj[0] != token_names_adj[1]

            data.append((
                prompt_idx,
                synthetic_prompt,
                token_names_adj,
                token_counts
            ))

    cache_file = Path("data/") / (Path(path).stem + ".yaml")
    cache_file.parent.mkdir(parents=True, exist_ok=True)

    with open(cache_file, "w") as f:
        yaml.dump(data, f, Dumper=yaml.CSafeDumper)

    return data


def run_configs(configs, out_paths, data, gpu_id=None):
    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

    stable = setup()

    for config, out_path in zip(configs, out_paths):
        run_all(stable, data, config, out_path)


def split_list(l, n):
    split_len = ceil(len(l) / n)
    
    return [l[i * split_len:min(((i + 1) * split_len), len(l))] for i in range(n)]


def test_counting(data, config_file, single_only=False, gpus=None):
    # data = load_hrs_data("../attention-refocusing/data_evaluate_LLM/HRS/counting_prompts.csv")
    
    with open(data, "r") as f:
        data = yaml.safe_load(f)

    if single_only:
        data = [(prompt_idx, prompt, token_names, token_counts) for (prompt_idx, prompt, token_names, token_counts) in data if len(token_names) == 1]

    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    if isinstance(config["seed"], list):
        configs, out_paths = [], []

        for seed in config["seed"]:
            configs.append({**config, "seed": seed})
            out_paths.append(Path("exp") / Path(config_file).stem / f"seed_{seed}")
    elif isinstance(config.get("counting_loss_scales", None), list) and isinstance(config["counting_loss_scales"][0], list):
        configs, out_paths = [], []

        for counting_loss_scales in config["counting_loss_scales"]:
            configs.append({**config, "counting_loss_scales": counting_loss_scales})
            out_paths.append(Path("exp") / Path(config_file).stem / f"cls_{','.join([str(c) for c in counting_loss_scales])}")
    elif isinstance(config.get("attention_loss_scale", None), list):
        configs, out_paths = [], []

        for attention_loss_scale in config["attention_loss_scale"]:
            configs.append({**config, "attention_loss_scale": attention_loss_scale})
            out_paths.append(Path("exp") / Path(config_file).stem / f"als_{attention_loss_scale}")
    else:
        configs = [config]
        out_paths = [Path("exp") / Path(config_file).stem]

    for config in configs:
        if config.get("attention_loss_l4", None) is not None:
            config["attention_loss_factor_scales"] = [None, None, config.get("attention_loss_l4")]
            del config["attention_loss_l4"]

    # config = dict(
    #     seed = 0,
    #     counting_loss_scales = [40],
    #     attention_loss_start = 25,
    #     attention_loss_scale = 5,
    # )

    if gpus is not None:
        num_workers = len(gpus)

        procs = []
        for i, (configs_split, out_paths_split) in enumerate(zip(split_list(configs, num_workers), split_list(out_paths, num_workers))):
            # mp.set_start_method('spawn')
            p = mp.Process(target=run_configs, args=(configs_split, out_paths_split, data, gpus[i]))
            procs.append(p)

        for p in procs:
            p.start()

        for p in procs:
            p.join()
    else:
        run_configs(configs, out_paths, data)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/obj_v1_prompts.yaml")
    parser.add_argument("--cfg")
    parser.add_argument("--single_cls_only", action="store_true")
    parser.add_argument("--gpus", nargs="+")
    args = parser.parse_args()

    test_counting(data=args.data, config_file=args.cfg, single_only=args.single_cls_only, gpus=args.gpus)


def main2():
    stable = setup()
    demo_single(stable)


if __name__ == "__main__":
    main()
