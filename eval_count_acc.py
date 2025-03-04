import json
import subprocess
import argparse
from pathlib import Path
import cv2
from typing import Any
import pickle
import yaml
import numpy as np

from tqdm import tqdm

import groundingdino
from groundingdino.util.inference import load_model, load_image, predict, annotate, preprocess_caption


class GroundingDINODetector:
    def __init__(self) -> None:
        self.model = load_model(
            str(Path(groundingdino.__file__).parent / "config/GroundingDINO_SwinT_OGC.py"), 
            "weights/groundingdino_swint_ogc.pth"
        )
        self.box_thres = 0.35
        self.text_thres = 0.25

    def __call__(self, img_path, classes) -> Any:
        image_source, image = load_image(img_path)

        prompt = " . ".join(classes)

        boxes, logits, phrases = predict(
            model=self.model,
            image=image,
            caption=prompt,
            box_threshold=self.box_thres,
            text_threshold=self.text_thres
        )

        return image_source, (boxes, logits, phrases) 

    def annote(self, image_source, res):
        (boxes, logits, phrases) = res
        annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
        return annotated_frame


def detect_grounding_dino(model, imgs, classes, save_path, save_vis=False):
    save_path = Path(save_path)
    save_file = save_path / "res.pkl"

    if save_file.is_file():
        print(f"loading cache {save_file}")
        with open(save_file, "rb") as f:
            return pickle.load(f)

    save_path.mkdir(parents=True, exist_ok=True)
    if save_vis:
        save_img_dir = save_path / "vis"
        save_img_dir.mkdir(exist_ok=True)

    res_all = {}
    for i, (img_path, cls) in enumerate(zip(tqdm(imgs), classes)):
        img, res = model(img_path, cls)

        res_all[img_path.stem] = res

        if save_vis:
            img = model.annote(img, res)
            cv2.imwrite(str(save_img_dir / f"{i}.jpg"), img,)
        
    with open(save_file, "wb") as f:
        pickle.dump(res_all, f)

    return res_all


def _count_list_items(l):
    res = {}

    for v in l:
        if v not in res:
            res[v] = 0
        res[v] += 1

    return res


class CountMetricF1:
    def __init__(self) -> None:
        self.true_pos = 0
        self.false_pos = 0
        self.false_neg = 0

    def add(self, pred_num, gt_num):
        self.true_pos += min(gt_num, pred_num)
        self.false_pos += max((pred_num-gt_num), 0)
        self.false_neg += max((gt_num-pred_num), 0)

    def compute(self):
        if self.true_pos + self.false_pos == 0:
            precision, recall = 0, 0
        else:
            precision = self.true_pos / (self.true_pos + self.false_pos)
            recall = self.true_pos / (self.true_pos + self.false_neg)

        f1 = ((2*precision*recall)/(precision+recall)) if precision+recall > 0 else 0

        return {
            "f1": f1,
            "precision": precision,
            "recall": recall
        }


class CountMetricHisto:
    def __init__(self) -> None:
        self.histo = {}

    def add(self, pred_num, gt_num):
        if gt_num not in self.histo:
            self.histo[gt_num] = []

        self.histo[gt_num].append(pred_num)

    def compute(self):
        res = [None] * (max(self.histo.keys()) + 1)

        for count, preds in self.histo.items():
            res[count] = {"mean": np.mean(preds), "min": np.min(preds), "max": np.max(preds)}

        return res
    

class CountMetricMAE:
    def __init__(self, normalize=True) -> None:
        self.normalize = normalize
        self.diffs = []

    def add(self, pred_num, gt_num):
        diff = pred_num - gt_num

        if self.normalize:
            diff /= gt_num

        self.diffs.append(diff)
        return diff
    
    def add_from_metric(self, metric):
        assert isinstance(metric, CountMetricMAE) and metric.normalize == self.normalize

        self.diffs += metric.diffs

    def compute_loss(self):
        n = len(self.diffs)
        mae = np.sum(np.abs(self.diffs)) / n
        return mae

    def compute(self):
        n = len(self.diffs)

        mae = np.sum(np.abs(self.diffs)) / n
        rmse = np.sqrt(np.sum(np.square(self.diffs)) / n)

        return {"mae": mae, "rmse": rmse}


def compute_metric(metrics, classes_pred, classes_gt):
    for prompt, gt in classes_gt.items():
        try:
            pred = classes_pred[prompt]
        except KeyError:
            print(f"Prompt {prompt} not found in predictions")
            continue

        for cls_name, gt_num in gt.items():
            pred_num = pred.get(cls_name, 0)

            for metric in metrics:
                metric.add(pred_num, gt_num)

    return [metric.compute() for metric in metrics]


def compute_metric_pick_best(metric_cls, classed_pred_all, classes_gt):
    metric_best = metric_cls()

    for prompt, gt in classes_gt.items():
        preds = [classes_pred[prompt] for classes_pred in classed_pred_all]

        metrics_local = []
        for pred in preds:
            metric_local = metric_cls()
            for cls_name, gt_num in gt.items():
                pred_num = pred.get(cls_name, 0)
                metric_local.add(pred_num, gt_num)
            metrics_local.append(metric_local)
        
        metric_best_idx = np.argmin([m.compute_loss() for m in metrics_local])
        metric_best.add_from_metric(metrics_local[metric_best_idx])

    return metric_best.compute()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path")
    parser.add_argument("--data")
    parser.add_argument("--cls_exclude", nargs="*")
    args = parser.parse_args()

    if "*" in args.path:
        in_paths = sorted(Path().glob(args.path))
    else:
        in_paths = [Path(args.path)]

    with open(args.data, "r") as f:
        data = yaml.safe_load(f)

    img_classes = {d["prompt"]: d for d in data}
    classes_gt = {k: dict(zip(v["objs_singular"], v["counts"])) for k, v in img_classes.items()}

    classes_preds = []

    model = GroundingDINODetector()
    for in_path in in_paths:
        img_files = list(in_path.glob("*.png"))
        
        out_path = Path("eval") / in_path

        img_class_names = [img_classes[f.stem]["objs_singular"] for f in img_files]
        res_all = detect_grounding_dino(model, img_files, img_class_names, out_path, True)

        classes_pred = {k: _count_list_items(v[2]) for k, v in res_all.items()}
        classes_preds.append(classes_pred)

    # metric_res = compute_metric([CountMetricF1(), CountMetricHisto(), CountMetricMAE()], classes_pred, classes_gt)

    if args.cls_exclude is not None:
        excl = set(args.cls_exclude)
        prompt_del = [prompt for prompt, objs_counts in classes_gt.items() if len(excl & set(objs_counts.keys())) > 0]
        
        for p in prompt_del:
            del classes_gt[p]
            for pred in classes_preds:
                del pred[p]

    if len(classes_preds) == 1:
        metric_res = compute_metric([CountMetricMAE()], classes_preds[0], classes_gt)
    else:
        metric_res = compute_metric_pick_best(CountMetricMAE, classes_preds, classes_gt)

    print(metric_res)

    with open(Path(in_paths[0]) / "eval.json", "w") as f:
        json.dump(metric_res, f)

    with open("eval.txt", "a") as f:
        f.write(f"{args.path}: {metric_res}\n")


if __name__ == "__main__":
    main()
