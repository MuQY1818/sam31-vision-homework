"""Run SAM 3.1 text-prompt image segmentation cases."""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
import time
import warnings
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAM3_REPO = PROJECT_ROOT / "sam3"
if str(SAM3_REPO) not in sys.path:
    sys.path.insert(0, str(SAM3_REPO))

warnings.filterwarnings("ignore", message="pkg_resources is deprecated")
warnings.filterwarnings("ignore", category=FutureWarning, module="timm")

from sam3.model.sam3_image_processor import Sam3Processor  # noqa: E402
from sam3.model_builder import build_sam3_image_model  # noqa: E402


COLORS = [
    (64, 145, 255),
    (255, 92, 92),
    (63, 190, 128),
    (255, 190, 80),
    (176, 107, 255),
    (255, 116, 195),
]


@dataclass(frozen=True)
class ImageCase:
    image_path: Path
    prompt: str
    output_prefix: str
    note: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT / "checkpoints" / "sam3.1" / "sam3.1_multiplex.pt",
        help="SAM 3.1 multiplex checkpoint path.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=PROJECT_ROOT / "results",
        help="Directory for result images and metadata.",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=PROJECT_ROOT / "logs",
        help="Directory for model loading logs.",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.35,
        help="Confidence threshold used by Sam3Processor.",
    )
    parser.add_argument("--max-instances", type=int, default=8)
    return parser.parse_args()


def default_cases() -> list[ImageCase]:
    image_dir = PROJECT_ROOT / "data" / "static_images"
    return [
        ImageCase(
            image_path=image_dir / "cumt_nanhu_library.jpg",
            prompt="building",
            output_prefix="sam31_cumt_building",
            note="CUMT Nanhu campus library night scene.",
        ),
        ImageCase(
            image_path=image_dir / "cumt_nanhu_library.jpg",
            prompt="tree",
            output_prefix="sam31_cumt_tree",
            note="CUMT Nanhu campus vegetation and foreground objects.",
        ),
        ImageCase(
            image_path=image_dir / "groceries.jpg",
            prompt="bottle",
            output_prefix="sam31_groceries_bottle",
            note="Complex grocery shelf scene.",
        ),
        ImageCase(
            image_path=image_dir / "truck.jpg",
            prompt="truck",
            output_prefix="sam31_truck",
            note="Large vehicle object in a street scene.",
        ),
        ImageCase(
            image_path=image_dir / "truck.jpg",
            prompt="wheel",
            output_prefix="sam31_truck_wheel",
            note="Part-level prompt on the truck image.",
        ),
    ]


def as_numpy_masks(masks: torch.Tensor) -> list:
    if masks.numel() == 0:
        return []
    masks = masks.detach().cpu().bool()
    if masks.ndim == 4:
        masks = masks.squeeze(1)
    return [mask.numpy() for mask in masks]


def render_result(
    image_path: Path,
    output: dict,
    prompt: str,
    out_path: Path,
    max_instances: int,
) -> int:
    image = Image.open(image_path).convert("RGB")
    canvas = image.convert("RGBA")
    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font = ImageFont.load_default()

    masks = output.get("masks", torch.empty(0))
    boxes = output.get("boxes", torch.empty(0))
    scores = output.get("scores", torch.empty(0))
    mask_list = as_numpy_masks(masks)

    order = list(range(len(mask_list)))
    if len(scores) > 0:
        score_values = scores.detach().cpu().float().tolist()
        order.sort(key=lambda idx: score_values[idx], reverse=True)
    else:
        score_values = [0.0 for _ in mask_list]

    kept = order[:max_instances]
    for visible_idx, idx in enumerate(kept):
        color = COLORS[visible_idx % len(COLORS)]
        mask = mask_list[idx]
        mask_img = Image.new("RGBA", canvas.size, color + (0,))
        alpha = Image.fromarray((mask.astype("uint8") * 105), mode="L")
        mask_img.putalpha(alpha)
        overlay.alpha_composite(mask_img)

        if idx < len(boxes):
            x0, y0, x1, y1 = boxes[idx].detach().cpu().float().tolist()
            draw.rectangle([x0, y0, x1, y1], outline=color + (255,), width=3)
            draw.text(
                (x0 + 3, max(0, y0 - 14)),
                f"{prompt} {score_values[idx]:.2f}",
                fill=color + (255,),
                font=font,
            )

    canvas.alpha_composite(overlay)
    draw = ImageDraw.Draw(canvas)
    draw.rectangle([0, 0, canvas.width, 28], fill=(0, 0, 0, 130))
    draw.text(
        (8, 8),
        f"prompt: {prompt} | objects: {len(mask_list)}",
        fill=(255, 255, 255, 255),
        font=font,
    )
    canvas.convert("RGB").save(out_path)
    return len(mask_list)


def create_contact_sheet(image_paths: list[Path], output_path: Path) -> None:
    thumbs = []
    thumb_width = 360
    caption_height = 36
    for path in image_paths:
        image = Image.open(path).convert("RGB")
        ratio = thumb_width / image.width
        thumb_height = int(image.height * ratio)
        image = image.resize((thumb_width, thumb_height), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (thumb_width, thumb_height + caption_height), "white")
        canvas.paste(image, (0, 0))
        draw = ImageDraw.Draw(canvas)
        draw.text((8, thumb_height + 9), path.name, fill=(0, 0, 0))
        thumbs.append(canvas)

    if not thumbs:
        return
    columns = min(3, len(thumbs))
    rows = (len(thumbs) + columns - 1) // columns
    sheet_width = columns * thumb_width
    sheet_height = rows * max(thumb.height for thumb in thumbs)
    sheet = Image.new("RGB", (sheet_width, sheet_height), "white")
    for idx, thumb in enumerate(thumbs):
        x = (idx % columns) * thumb_width
        y = (idx // columns) * thumb.height
        sheet.paste(thumb, (x, y))
    sheet.save(output_path)


def main() -> None:
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)

    if not args.checkpoint.exists():
        raise FileNotFoundError(args.checkpoint)
    for case in default_cases():
        if not case.image_path.exists():
            raise FileNotFoundError(case.image_path)

    load_log = args.log_dir / "sam31_image_cases_model_load.log"
    start_time = time.perf_counter()
    with load_log.open("w", encoding="utf-8") as log_file:
        with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
            model = build_sam3_image_model(
                checkpoint_path=str(args.checkpoint),
                load_from_HF=False,
                compile=False,
            )
            processor = Sam3Processor(
                model,
                confidence_threshold=args.confidence_threshold,
            )
    load_seconds = time.perf_counter() - start_time

    rendered_paths: list[Path] = []
    cases_metadata = []
    autocast_context = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if torch.cuda.is_available()
        else nullcontext()
    )
    for case in default_cases():
        image = Image.open(case.image_path).convert("RGB")
        start_time = time.perf_counter()
        with autocast_context:
            state = processor.set_image(image)
            output = processor.set_text_prompt(state=state, prompt=case.prompt)
        inference_seconds = time.perf_counter() - start_time

        out_path = args.results_dir / f"{case.output_prefix}.png"
        object_count = render_result(
            image_path=case.image_path,
            output=output,
            prompt=case.prompt,
            out_path=out_path,
            max_instances=args.max_instances,
        )
        rendered_paths.append(out_path)

        scores = output.get("scores", torch.empty(0)).detach().cpu().float().tolist()
        cases_metadata.append(
            {
                "image": str(case.image_path),
                "prompt": case.prompt,
                "output_prefix": case.output_prefix,
                "note": case.note,
                "objects": object_count,
                "inference_seconds": round(inference_seconds, 2),
                "top_scores": [round(score, 3) for score in scores[: args.max_instances]],
                "rendered_image": str(out_path),
            }
        )

    summary_path = args.results_dir / "sam31_image_cases_summary.png"
    create_contact_sheet(rendered_paths, summary_path)

    metadata = {
        "project": "facebookresearch/sam3",
        "modelscope_model": "facebook/sam3.1",
        "checkpoint": str(args.checkpoint),
        "load_seconds": round(load_seconds, 2),
        "confidence_threshold": args.confidence_threshold,
        "max_instances": args.max_instances,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_runtime": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cases": cases_metadata,
        "summary_image": str(summary_path),
        "load_log": str(load_log),
    }
    metadata_path = args.results_dir / "sam31_image_cases_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
