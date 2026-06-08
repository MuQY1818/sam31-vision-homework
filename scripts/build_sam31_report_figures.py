"""Build polished SAM 3.1 figures for the LaTeX report."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_RESULTS_DIR = PROJECT_ROOT / "results"
VIDEO_RESULTS_DIR = (
    RAW_RESULTS_DIR / "video"
    if (RAW_RESULTS_DIR / "video").exists()
    else RAW_RESULTS_DIR
)
IMAGE_RESULTS_DIR = (
    RAW_RESULTS_DIR / "image"
    if (RAW_RESULTS_DIR / "image").exists()
    else RAW_RESULTS_DIR
)
FRAMES_DIR = PROJECT_ROOT / "data" / "player_frames"
FIGURES_DIR = (
    PROJECT_ROOT / "report" / "sam31_analysis" / "figures"
    if (PROJECT_ROOT / "report" / "sam31_analysis").exists()
    else PROJECT_ROOT / "report" / "source" / "figures"
)

FONT_DIR = Path("/usr/share/fonts/opentype/noto")
FONT_REGULAR = FONT_DIR / "NotoSansCJK-Regular.ttc"
FONT_MEDIUM = FONT_DIR / "NotoSansCJK-Medium.ttc"
FONT_BOLD = FONT_DIR / "NotoSansCJK-Bold.ttc"
FONT_INDEX = 2

INK = (33, 37, 43)
MUTED = (93, 101, 113)
LINE = (213, 220, 230)
SOFT_LINE = (229, 234, 241)
PAPER = (248, 250, 252)
WHITE = (255, 255, 255)
BLUE = (11, 91, 168)
TEAL = (20, 126, 113)
RED = (175, 49, 45)
AMBER = (181, 112, 16)


@dataclass(frozen=True)
class VideoRow:
    prompt: str
    descriptor: str
    count: str
    prefix: str
    accent: tuple[int, int, int]


@dataclass(frozen=True)
class ImageCard:
    title: str
    subtitle: str
    image_path: Path
    accent: tuple[int, int, int]


def font(path: Path, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(path), size=size, index=FONT_INDEX)


def text_size(
    draw: ImageDraw.ImageDraw,
    text: str,
    text_font: ImageFont.FreeTypeFont,
) -> tuple[int, int]:
    left, top, right, bottom = draw.textbbox((0, 0), text, font=text_font)
    return right - left, bottom - top


def draw_centered_text(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    text: str,
    text_font: ImageFont.FreeTypeFont,
    fill: tuple[int, int, int],
) -> None:
    width, height = text_size(draw, text, text_font)
    x0, y0, x1, y1 = box
    x = x0 + (x1 - x0 - width) // 2
    y = y0 + (y1 - y0 - height) // 2 - 1
    draw.text((x, y), text, font=text_font, fill=fill)


def fit_cover(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    target_width, target_height = size
    source_ratio = image.width / image.height
    target_ratio = target_width / target_height

    if source_ratio > target_ratio:
        new_height = target_height
        new_width = round(new_height * source_ratio)
    else:
        new_width = target_width
        new_height = round(new_width / source_ratio)

    resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    left = (new_width - target_width) // 2
    top = (new_height - target_height) // 2
    return resized.crop((left, top, left + target_width, top + target_height))


def paste_rounded(
    canvas: Image.Image,
    image: Image.Image,
    xy: tuple[int, int],
    radius: int = 8,
) -> None:
    mask = Image.new("L", image.size, 0)
    mask_draw = ImageDraw.Draw(mask)
    mask_draw.rounded_rectangle(
        (0, 0, image.width - 1, image.height - 1),
        radius=radius,
        fill=255,
    )
    canvas.paste(image, xy, mask)


def clean_video_result(result_path: Path, frame_path: Path) -> Image.Image:
    result = Image.open(result_path).convert("RGB")
    if not frame_path.exists():
        raise FileNotFoundError(frame_path)

    # The rendered result contains a white "Frame X" debug label in the top-left.
    frame = Image.open(frame_path).convert("RGB").resize(result.size)
    patch = frame.crop((0, 0, 215, 56))
    result.paste(patch, (0, 0))
    return result


def draw_chip(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    fill: tuple[int, int, int],
    text_font: ImageFont.FreeTypeFont,
) -> None:
    x, y = xy
    width, height = text_size(draw, text, text_font)
    pad_x = 14
    pad_y = 7
    draw.rounded_rectangle(
        (x, y, x + width + pad_x * 2, y + height + pad_y * 2),
        radius=8,
        fill=fill,
    )
    draw.text((x + pad_x, y + pad_y - 1), text, font=text_font, fill=WHITE)


def draw_video_grid() -> None:
    rows = [
        VideoRow("person", "粗粒度类别", "每帧 3 个球员", "sam31_player_person", BLUE),
        VideoRow("ball", "小目标提示", "每帧 1 个足球", "sam31_player_ball", TEAL),
        VideoRow("player in red", "属性约束", "每帧 1 名红衣球员", "sam31_player_red", RED),
    ]
    frame_ids = [0, 4, 7]
    output_path = FIGURES_DIR / "sam31-video-prompt-grid.png"
    if not all((FRAMES_DIR / f"{frame_id:05d}.jpg").exists() for frame_id in frame_ids):
        if output_path.exists():
            return
        raise FileNotFoundError(
            f"Missing extracted player frames under {FRAMES_DIR}. "
            "Run scripts/run_sam31_player.py before rebuilding the video grid."
        )

    margin_x = 42
    margin_y = 34
    label_width = 244
    header_height = 52
    cell_width = 456
    cell_height = 257
    gap_x = 20
    gap_y = 22

    width = margin_x * 2 + label_width + gap_x + len(frame_ids) * cell_width
    width += (len(frame_ids) - 1) * gap_x
    height = margin_y * 2 + header_height + len(rows) * cell_height
    height += (len(rows) - 1) * gap_y

    canvas = Image.new("RGB", (width, height), PAPER)
    draw = ImageDraw.Draw(canvas)
    title_font = font(FONT_BOLD, 32)
    body_font = font(FONT_REGULAR, 25)
    small_font = font(FONT_REGULAR, 21)
    header_font = font(FONT_MEDIUM, 25)

    table_x = margin_x
    table_y = margin_y + header_height
    image_x = table_x + label_width + gap_x

    for col, frame_id in enumerate(frame_ids):
        x = image_x + col * (cell_width + gap_x)
        draw_centered_text(
            draw,
            (x, margin_y + 5, x + cell_width, margin_y + header_height - 7),
            f"第 {frame_id} 帧",
            header_font,
            BLUE,
        )

    for row_index, row in enumerate(rows):
        y = table_y + row_index * (cell_height + gap_y)
        draw.rounded_rectangle(
            (table_x, y, table_x + label_width, y + cell_height),
            radius=8,
            fill=WHITE,
            outline=LINE,
            width=2,
        )
        draw.rounded_rectangle(
            (table_x, y, table_x + 9, y + cell_height),
            radius=8,
            fill=row.accent,
        )
        draw.text((table_x + 24, y + 58), row.prompt, font=title_font, fill=row.accent)
        draw.text((table_x + 24, y + 116), row.descriptor, font=body_font, fill=MUTED)
        draw.text((table_x + 24, y + 163), row.count, font=small_font, fill=MUTED)

        for col, frame_id in enumerate(frame_ids):
            x = image_x + col * (cell_width + gap_x)
            result_path = VIDEO_RESULTS_DIR / f"{row.prefix}_frame_{frame_id:05d}.png"
            frame_path = FRAMES_DIR / f"{frame_id:05d}.jpg"
            image = clean_video_result(result_path, frame_path)
            image = fit_cover(image, (cell_width, cell_height))

            shadow = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
            shadow_draw = ImageDraw.Draw(shadow)
            shadow_draw.rounded_rectangle(
                (x + 2, y + 3, x + cell_width + 2, y + cell_height + 3),
                radius=8,
                fill=(30, 41, 59, 18),
            )
            canvas.paste(Image.alpha_composite(canvas.convert("RGBA"), shadow).convert("RGB"))
            paste_rounded(canvas, image, (x, y), radius=8)
            draw.rounded_rectangle(
                (x, y, x + cell_width, y + cell_height),
                radius=8,
                outline=SOFT_LINE,
                width=2,
            )

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=95)


def draw_image_card(
    canvas: Image.Image,
    xy: tuple[int, int],
    card: ImageCard,
    size: tuple[int, int],
) -> None:
    draw = ImageDraw.Draw(canvas)
    x, y = xy
    width, height = size
    image_height = 360
    footer_y = y + image_height
    title_font = font(FONT_BOLD, 34)
    subtitle_font = font(FONT_REGULAR, 25)
    chip_font = font(FONT_MEDIUM, 20)

    draw.rounded_rectangle(
        (x, y, x + width, y + height),
        radius=8,
        fill=WHITE,
        outline=LINE,
        width=2,
    )
    image = Image.open(card.image_path).convert("RGB")
    image = fit_cover(image, (width, image_height))
    paste_rounded(canvas, image, (x, y), radius=8)
    draw.rectangle((x, footer_y - 7, x + width, footer_y + 2), fill=WHITE)
    draw.rectangle((x, footer_y, x + width, y + height), fill=WHITE)
    draw.rounded_rectangle((x, y, x + width, y + height), radius=8, outline=LINE, width=2)

    draw.text((x + 28, footer_y + 24), card.title, font=title_font, fill=card.accent)
    draw.text((x + 28, footer_y + 73), card.subtitle, font=subtitle_font, fill=MUTED)
    draw_chip(draw, (x + width - 148, footer_y + 32), "SAM 3.1", card.accent, chip_font)


def draw_failure_card(
    canvas: Image.Image,
    xy: tuple[int, int],
    size: tuple[int, int],
) -> None:
    draw = ImageDraw.Draw(canvas)
    x, y = xy
    width, height = size
    title_font = font(FONT_BOLD, 32)
    label_font = font(FONT_BOLD, 25)
    body_font = font(FONT_REGULAR, 23)
    small_font = font(FONT_MEDIUM, 19)

    draw.rounded_rectangle(
        (x, y, x + width, y + height),
        radius=8,
        fill=WHITE,
        outline=LINE,
        width=2,
    )
    draw.text((x + 28, y + 26), "阈值敏感提示", font=title_font, fill=AMBER)
    draw.text(
        (x + 28, y + 76),
        "在 confidence=0.35 下，两组 prompt 未保留实例。",
        font=body_font,
        fill=MUTED,
    )

    thumb_y = y + 125
    thumb_width = (width - 72) // 2
    thumb_height = 206
    samples = [
        ("tree", IMAGE_RESULTS_DIR / "sam31_cumt_tree.png"),
        ("bottle", IMAGE_RESULTS_DIR / "sam31_groceries_bottle.png"),
    ]
    for idx, (label, image_path) in enumerate(samples):
        tx = x + 28 + idx * (thumb_width + 16)
        image = Image.open(image_path).convert("RGB")
        image = fit_cover(image, (thumb_width, thumb_height))
        paste_rounded(canvas, image, (tx, thumb_y), radius=8)
        draw.rounded_rectangle(
            (tx, thumb_y, tx + thumb_width, thumb_y + thumb_height),
            radius=8,
            outline=SOFT_LINE,
            width=2,
        )
        draw.rounded_rectangle(
            (tx + 12, thumb_y + 12, tx + 115, thumb_y + 45),
            radius=8,
            fill=(255, 255, 255),
            outline=LINE,
            width=1,
        )
        draw.text((tx + 24, thumb_y + 18), f"{label}: 0", font=small_font, fill=AMBER)

    text_y = thumb_y + thumb_height + 27
    draw.text((x + 28, text_y), "tree", font=label_font, fill=INK)
    draw.text(
        (x + 118, text_y + 1),
        "夜景树木与暗背景混合，候选被阈值过滤。",
        font=body_font,
        fill=MUTED,
    )
    draw.text((x + 28, text_y + 52), "bottle", font=label_font, fill=INK)
    draw.text(
        (x + 118, text_y + 53),
        "纸袋遮挡目标外观，未形成稳定候选。",
        font=body_font,
        fill=MUTED,
    )


def draw_image_grid() -> None:
    margin = 42
    gap = 28
    card_width = 780
    card_height = 470
    width = margin * 2 + card_width * 2 + gap
    height = margin * 2 + card_height * 2 + gap
    canvas = Image.new("RGB", (width, height), PAPER)

    cards = [
        ImageCard(
            "building",
            "校园夜景，保留 11 个建筑区域",
            IMAGE_RESULTS_DIR / "sam31_cumt_building.png",
            BLUE,
        ),
        ImageCard(
            "truck",
            "整车提示，输出 1 个完整实例",
            IMAGE_RESULTS_DIR / "sam31_truck.png",
            TEAL,
        ),
        ImageCard(
            "wheel",
            "部件级提示，输出 4 个车轮相关区域",
            IMAGE_RESULTS_DIR / "sam31_truck_wheel.png",
            RED,
        ),
    ]

    positions = [
        (margin, margin),
        (margin + card_width + gap, margin),
        (margin, margin + card_height + gap),
    ]
    for card, position in zip(cards, positions):
        draw_image_card(canvas, position, card, (card_width, card_height))

    draw_failure_card(
        canvas,
        (margin + card_width + gap, margin + card_height + gap),
        (card_width, card_height),
    )

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    canvas.save(FIGURES_DIR / "sam31-image-cases-grid.png", quality=95)


def main() -> None:
    draw_video_grid()
    draw_image_grid()


if __name__ == "__main__":
    main()
