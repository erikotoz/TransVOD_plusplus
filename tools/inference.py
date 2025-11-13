#!/usr/bin/env python3
"""Utility for running TransVOD++ inference and visualizing detections."""
from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Dict, List

import torch
from PIL import Image, ImageDraw, ImageFont

from datasets import build_dataset
from main import get_args_parser
from models import build_model

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225])


def parse_args() -> argparse.Namespace:
    """Build an argument parser that reuses the training defaults."""
    parent = get_args_parser()
    parser = argparse.ArgumentParser(
        "TransVOD++ inference script", parents=[parent], add_help=True
    )
    parser.add_argument(
        "--checkpoint_path",
        required=True,
        type=str,
        help="Path to the checkpoint that should be used for inference.",
    )
    parser.add_argument(
        "--image_set",
        default="val",
        choices=["train_vid", "train_det", "train_joint", "val"],
        help="Dataset split to draw samples from.",
    )
    parser.add_argument(
        "--sample_index",
        default=0,
        type=int,
        help="Index of the sample that should be visualized. Use a negative value to sample randomly.",
    )
    parser.add_argument(
        "--confidence_threshold",
        default=0.5,
        type=float,
        help="Score threshold for rendering bounding boxes.",
    )
    parser.add_argument(
        "--output_path",
        default="inference_result.png",
        type=str,
        help="Destination path for the visualization.",
    )
    parser.add_argument(
        "--seed",
        default=42,
        type=int,
        help="Seed for selecting a random sample when --sample_index is negative.",
    )
    return parser.parse_args()


def select_device(device_str: str) -> torch.device:
    if device_str.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA was requested but no GPU is available. Falling back to CPU.")
        return torch.device("cpu")
    return torch.device(device_str)


def move_to_device(target: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in target.items()}


def get_utils_module(dataset_file: str):
    if dataset_file == "vid_single":
        import util.misc as utils
    else:
        import util.misc_multi as utils
    return utils


def build_nested_tensor(sample: torch.Tensor, dataset_file: str, device: torch.device):
    utils = get_utils_module(dataset_file)
    nested = utils.nested_tensor_from_tensor_list([sample])
    return nested.to(device)


def tensor_to_pil(frame: torch.Tensor) -> Image.Image:
    frame = frame.detach().cpu()
    mean = IMAGENET_MEAN[:, None, None]
    std = IMAGENET_STD[:, None, None]
    frame = frame * std + mean
    frame = frame.clamp(0, 1)
    array = (frame.permute(1, 2, 0).numpy() * 255).astype("uint8")
    return Image.fromarray(array)


def split_frames(sample: torch.Tensor) -> List[torch.Tensor]:
    if sample.dim() != 3 or sample.size(0) % 3 != 0:
        raise ValueError("Expected a CHW tensor where C is divisible by 3.")
    return list(sample.split(3, dim=0))


def get_category_names(dataset) -> Dict[int, str]:
    base = dataset
    from torch.utils.data import ConcatDataset

    if isinstance(base, ConcatDataset):
        base = base.datasets[0]
    if hasattr(base, "coco"):
        return {cat["id"]: cat["name"] for cat in base.coco.cats.values()}
    return {}


def draw_predictions(
    image: Image.Image,
    boxes: torch.Tensor,
    labels: torch.Tensor,
    scores: torch.Tensor,
    id_to_name: Dict[int, str],
    threshold: float,
) -> Image.Image:
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    palette = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
    ]
    for box, label, score in zip(boxes, labels, scores):
        score_value = float(score)
        if score_value < threshold:
            continue
        coords = box.tolist()
        label_id = int(label)
        color = palette[label_id % len(palette)]
        label_name = id_to_name.get(label_id, f"class_{label_id}")
        text = f"{label_name}: {score_value:.2f}"
        draw.rectangle(coords, outline=color, width=3)
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        text_origin = (coords[0], max(0, coords[1] - text_height - 4))
        draw.rectangle(
            [text_origin, (text_origin[0] + text_width + 4, text_origin[1] + text_height + 4)],
            fill=color,
        )
        draw.text((text_origin[0] + 2, text_origin[1] + 2), text, fill="white", font=font)
    return image


def maybe_select_random_index(args, dataset_length: int) -> int:
    if args.sample_index >= 0:
        return args.sample_index
    random.seed(args.seed)
    return random.randint(0, dataset_length - 1)


def main():
    args = parse_args()
    args.distributed = False
    args.eval = True

    checkpoint_path = Path(args.checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint {checkpoint_path} does not exist.")

    device = select_device(args.device)
    model, _, postprocessors = build_model(args)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    missing_keys, unexpected_keys = model.load_state_dict(checkpoint["model"], strict=False)
    if missing_keys:
        print(f"Missing keys when loading the checkpoint: {missing_keys}")
    if unexpected_keys:
        print(f"Unexpected keys when loading the checkpoint: {unexpected_keys}")
    model.to(device)
    model.eval()

    dataset = build_dataset(image_set=args.image_set, args=args)
    sample_index = maybe_select_random_index(args, len(dataset))
    sample, target = dataset[sample_index]
    frames = split_frames(sample)
    nested = build_nested_tensor(sample, args.dataset_file, device)
    target = move_to_device(target, device)

    with torch.no_grad():
        outputs = model(nested)
    orig_target_sizes = torch.stack([target["orig_size"]]).to(device)
    processed = postprocessors["bbox"](outputs, orig_target_sizes)[0]

    id_to_name = get_category_names(dataset)
    key_frame = tensor_to_pil(frames[0])
    visualization = draw_predictions(
        key_frame,
        processed["boxes"].cpu(),
        processed["labels"].cpu(),
        processed["scores"].cpu(),
        id_to_name,
        args.confidence_threshold,
    )

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    visualization.save(output_path)

    image_id = int(target["image_id"].item()) if isinstance(target["image_id"], torch.Tensor) else target["image_id"]
    kept = processed["scores"] >= args.confidence_threshold
    print(f"Saved visualization for sample #{sample_index} (image id {image_id}) to {output_path}.")
    print("Detections above the threshold:")
    for box, label, score in zip(
        processed["boxes"][kept], processed["labels"][kept], processed["scores"][kept]
    ):
        label_id = int(label)
        label_name = id_to_name.get(label_id, f"class_{label_id}")
        print(f"  - {label_name} (id={label_id}) score={float(score):.3f} box={box.tolist()}")


if __name__ == "__main__":
    main()
