import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List

import clip
import numpy as np
import pyrallis
import torch
from PIL import Image
from tqdm import tqdm

sys.path.append(".")
sys.path.append("..")

from metrics.imagenet_utils import get_embedding_for_prompt, imagenet_templates


@dataclass
class EvalConfig:
    output_path: List[str] = None
    metrics_save_path: List[Path] = None  # Path("./metrics/")

    def __post_init__(self):
        output_path_eval = []

        for path in self.output_path:
            if "*" in path:
                output_path_eval += [p for p in Path().glob(str(path)) if p.is_dir()]
            else:
                output_path_eval += [path]

        self.output_path = output_path_eval
        self.output_path = [Path(p) for p in self.output_path]

        if self.metrics_save_path is None:
            self.metrics_save_path = self.output_path

        for d in self.metrics_save_path:
            d.mkdir(parents=True, exist_ok=True)


@pyrallis.wrap()
def run(config: EvalConfig):
    print("Loading CLIP model...")
    device = torch.device("cuda" if (torch.cuda.is_available() and torch.cuda.device_count() > 0) else "cpu")
    model, preprocess = clip.load("ViT-B/16", device)
    model.eval()

    for output_path, metrics_save_path in zip(config.output_path, config.metrics_save_path):
        prompts = [p.stem for p in output_path.glob("*.png")]
        print(f"Running on {len(prompts)} prompts...")

        results_per_prompt = {}
        for prompt in tqdm(prompts):

            print(f'Running on: "{prompt}"')

            # get all images for the given prompt
            # image_paths = [p for p in (config.output_path / prompt).rglob('*') if p.suffix in ['.png', '.jpg']]
            image_paths = [output_path / (prompt + ".png")]
            images = [Image.open(p) for p in image_paths]
            image_names = [p.name for p in image_paths]
            queries = [preprocess(image).unsqueeze(0).to(device) for image in images]

            with torch.no_grad():

                # split prompt into first and second halves
                if ' and ' in prompt:
                    prompt_parts = prompt.split(' and ')
                elif ' with ' in prompt:
                    prompt_parts = prompt.split(' with ')
                else:
                    prompt_parts = [prompt]
                    # print(f"Unable to split prompt: {prompt}. "
                    #     f"Looking for 'and' or 'with' for splitting! Skipping!")
                    # continue

                # extract texture features
                full_text_features = get_embedding_for_prompt(model, prompt, templates=imagenet_templates)
                first_half_features = get_embedding_for_prompt(model, prompt_parts[0], templates=imagenet_templates)
                second_half_features = get_embedding_for_prompt(model, prompt_parts[1], templates=imagenet_templates) if len(prompt_parts) > 1 else None

                # extract image features
                images_features = [model.encode_image(image) for image in queries]
                images_features = [feats / feats.norm(dim=-1, keepdim=True) for feats in images_features]

                # compute similarities
                full_text_similarities = [(feat.float() @ full_text_features.T).item() for feat in images_features]
                first_half_similarities = [(feat.float() @ first_half_features.T).item() for feat in images_features]
                second_half_similarities = [(feat.float() @ second_half_features.T).item() for feat in images_features] if second_half_features is not None else None

                results_per_prompt[prompt] = {
                    'full_text': full_text_similarities,
                    'first_half': first_half_similarities,
                    'second_half': second_half_similarities,
                    'image_names': image_names,
                }

        # aggregate results
        aggregated_results = {
            'full_text_aggregation': aggregate_by_full_text(results_per_prompt),
            'min_first_second_aggregation': aggregate_by_min_half(results_per_prompt),
        }

        with open(metrics_save_path / "clip_raw_metrics.json", 'w') as f:
            json.dump(results_per_prompt, f, sort_keys=True, indent=4)
        with open(metrics_save_path / "clip_aggregated_metrics.json", 'w') as f:
            json.dump(aggregated_results, f, sort_keys=True, indent=4)

        print("clip", aggregated_results)


def aggregate_by_min_half(d):
    """ Aggregate results for the minimum similarity score for each prompt. """

    if next(iter(d.values()))["second_half"] is None:
        return 0

    min_per_half_res = [[min(a, b) for a, b in zip(d[prompt]["first_half"], d[prompt]["second_half"])] for prompt in d]
    min_per_half_res = np.array(min_per_half_res).flatten()
    return np.average(min_per_half_res)


def aggregate_by_full_text(d):
    """ Aggregate results for the full text similarity for each prompt. """
    full_text_res = [v['full_text'] for v in d.values()]
    full_text_res = np.array(full_text_res).flatten()
    return np.average(full_text_res)


if __name__ == '__main__':
    run()
