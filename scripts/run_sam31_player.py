"""Run a small SAM 3.1 Object Multiplex video segmentation experiment."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import sys
import time
import uuid
import warnings
from pathlib import Path

import torch
from PIL import Image, ImageDraw, ImageFont, ImageSequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAM3_REPO = PROJECT_ROOT / "sam3"
if str(SAM3_REPO) not in sys.path:
    sys.path.insert(0, str(SAM3_REPO))

warnings.filterwarnings("ignore", message="pkg_resources is deprecated")
warnings.filterwarnings("ignore", category=FutureWarning, module="timm")

from sam3.model_builder import build_sam3_multiplex_video_predictor  # noqa: E402
from sam3.visualization_utils import save_masklet_image  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-gif",
        type=Path,
        default=SAM3_REPO / "assets" / "player.gif",
        help="Source GIF or image sequence used as the demo video.",
    )
    parser.add_argument(
        "--frames-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "player_frames",
        help="Directory for extracted JPEG frames.",
    )
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
        help="Directory for verbose model loading logs.",
    )
    parser.add_argument("--prompt", default="person", help="Text prompt.")
    parser.add_argument(
        "--output-prefix",
        default=None,
        help="Output filename prefix. Defaults to a prompt-based slug.",
    )
    parser.add_argument("--max-frames", type=int, default=12)
    parser.add_argument("--frame-step", type=int, default=4)
    return parser.parse_args()


def slugify_prompt(prompt: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", prompt.lower()).strip("_")
    return slug or "prompt"


def extract_gif_frames(
    source_gif: Path,
    frames_dir: Path,
    max_frames: int,
    step: int,
) -> list[Path]:
    frames_dir.mkdir(parents=True, exist_ok=True)
    for old_frame in frames_dir.glob("*.jpg"):
        old_frame.unlink()

    saved_frames: list[Path] = []
    with Image.open(source_gif) as gif:
        for idx, frame in enumerate(ImageSequence.Iterator(gif)):
            if idx % step != 0:
                continue
            frame_path = frames_dir / f"{len(saved_frames):05d}.jpg"
            frame.convert("RGB").save(frame_path, quality=95)
            saved_frames.append(frame_path)
            if len(saved_frames) >= max_frames:
                break

    if not saved_frames:
        raise RuntimeError(f"No frames extracted from {source_gif}")
    return saved_frames


def propagate_in_video(predictor, session_id: str) -> dict[int, dict]:
    outputs_per_frame = {}
    for response in predictor.handle_stream_request(
        request={"type": "propagate_in_video", "session_id": session_id}
    ):
        outputs_per_frame[response["frame_index"]] = response["outputs"]
    return outputs_per_frame


def start_multiplex_session(predictor, resource_path: Path) -> str:
    """Start a SAM3.1 multiplex session without unsupported base kwargs."""
    inference_state = predictor.model.init_state(
        resource_path=str(resource_path),
        offload_video_to_cpu=False,
        async_loading_frames=getattr(predictor, "async_loading_frames", False),
    )
    session_id = str(uuid.uuid4())
    predictor._all_inference_states[session_id] = {
        "state": inference_state,
        "session_id": session_id,
        "start_time": time.time(),
        "last_use_time": time.time(),
        "expiration_sec": getattr(predictor, "session_expiration_sec", 1200),
    }
    return session_id


def object_count(frame_output: dict) -> int:
    masks = frame_output["out_binary_masks"]
    return int(sum(bool(mask.any()) for mask in masks))


def create_contact_sheet(image_paths: list[Path], output_path: Path) -> None:
    images = [Image.open(path).convert("RGB") for path in image_paths]
    if not images:
        return

    thumb_width = 420
    caption_height = 34
    thumbs = []
    for path, image in zip(image_paths, images):
        ratio = thumb_width / image.width
        thumb_height = int(image.height * ratio)
        image = image.resize((thumb_width, thumb_height), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (thumb_width, thumb_height + caption_height), "white")
        canvas.paste(image, (0, 0))
        draw = ImageDraw.Draw(canvas)
        draw.text(
            (10, thumb_height + 8),
            path.name,
            fill=(0, 0, 0),
            font=ImageFont.load_default(),
        )
        thumbs.append(canvas)

    sheet_width = thumb_width * len(thumbs)
    sheet_height = max(thumb.height for thumb in thumbs)
    sheet = Image.new("RGB", (sheet_width, sheet_height), "white")
    for idx, thumb in enumerate(thumbs):
        sheet.paste(thumb, (idx * thumb_width, 0))
    sheet.save(output_path)


def main() -> None:
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    output_prefix = args.output_prefix or f"sam31_player_{slugify_prompt(args.prompt)}"

    if not args.checkpoint.exists():
        raise FileNotFoundError(args.checkpoint)
    if not args.source_gif.exists():
        raise FileNotFoundError(args.source_gif)

    frames = extract_gif_frames(
        source_gif=args.source_gif,
        frames_dir=args.frames_dir,
        max_frames=args.max_frames,
        step=args.frame_step,
    )

    load_log = args.log_dir / f"{output_prefix}_model_load.log"
    start_time = time.perf_counter()
    with load_log.open("w", encoding="utf-8") as log_file:
        with contextlib.redirect_stdout(log_file), contextlib.redirect_stderr(log_file):
            predictor = build_sam3_multiplex_video_predictor(
                checkpoint_path=str(args.checkpoint),
                use_fa3=False,
                compile=False,
                warm_up=False,
                async_loading_frames=False,
            )
            load_seconds = time.perf_counter() - start_time

            start_time = time.perf_counter()
            session_id = start_multiplex_session(predictor, args.frames_dir)
            response = predictor.handle_request(
                request={
                    "type": "add_prompt",
                    "session_id": session_id,
                    "frame_index": 0,
                    "text": args.prompt,
                }
            )
            outputs_per_frame = {0: response["outputs"]}
            outputs_per_frame.update(propagate_in_video(predictor, session_id))
            inference_seconds = time.perf_counter() - start_time

    selected_indices = sorted({0, len(frames) // 2, len(frames) - 1})
    rendered_paths = []
    for frame_idx in selected_indices:
        if frame_idx not in outputs_per_frame:
            continue
        output_path = args.results_dir / f"{output_prefix}_frame_{frame_idx:05d}.png"
        save_masklet_image(
            frame=str(frames[frame_idx]),
            outputs=outputs_per_frame[frame_idx],
            out_path=str(output_path),
            alpha=0.5,
            frame_idx=frame_idx,
        )
        rendered_paths.append(output_path)

    summary_path = args.results_dir / f"{output_prefix}_summary.png"
    create_contact_sheet(rendered_paths, summary_path)

    metadata = {
        "project": "facebookresearch/sam3",
        "modelscope_model": "facebook/sam3.1",
        "checkpoint": str(args.checkpoint),
        "prompt": args.prompt,
        "output_prefix": output_prefix,
        "source_gif": str(args.source_gif),
        "num_frames": len(frames),
        "selected_frames": selected_indices,
        "load_seconds": round(load_seconds, 2),
        "inference_seconds": round(inference_seconds, 2),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_runtime": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "objects_per_frame": {
            str(frame_idx): object_count(outputs)
            for frame_idx, outputs in sorted(outputs_per_frame.items())
        },
        "rendered_images": [str(path) for path in rendered_paths],
        "summary_image": str(summary_path),
        "load_log": str(load_log),
    }
    metadata_path = args.results_dir / f"{output_prefix}_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(metadata, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    main()
