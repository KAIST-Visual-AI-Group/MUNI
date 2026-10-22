"""Render and save media produced by the qualitative endpoint."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_PROJECT_FONTS_DIR = _PROJECT_ROOT / "fonts"
Font = ImageFont.ImageFont | ImageFont.FreeTypeFont

_FONT_CACHE: dict[tuple[str | None, int], Font] = {}


def _font(size: int, font_path: str | None = None) -> Font:
    key = (font_path, size)
    if key in _FONT_CACHE:
        return _FONT_CACHE[key]
    candidates: list[str] = []
    if font_path:
        candidates.append(font_path)
    if _PROJECT_FONTS_DIR.is_dir():
        for name in ("Inter.ttf", "NotoSans.ttf", "Roboto.ttf"):
            p = _PROJECT_FONTS_DIR / name
            if p.is_file():
                candidates.append(str(p))
    candidates.extend(
        [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/TTF/DejaVuSans.ttf",
        ]
    )
    for p in candidates:
        if Path(p).is_file():
            try:
                font = ImageFont.truetype(p, size=size)
                _FONT_CACHE[key] = font
                return font
            except Exception:
                continue
    font = ImageFont.load_default()
    _FONT_CACHE[key] = font
    return font


def _tensor_to_pil(image: torch.Tensor) -> Image.Image:
    """``(C, H, W)`` float in ``[0, 1]`` → PIL RGB."""
    arr = image.detach().cpu().float().clamp(0, 1).numpy()
    if arr.ndim == 3 and arr.shape[0] in (1, 3):
        arr = arr.transpose(1, 2, 0)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, axis=-1)
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    arr = np.round(arr * 255.0).astype(np.uint8)
    return Image.fromarray(arr)


def _text_width(draw: ImageDraw.ImageDraw, text: str, font) -> int:
    bbox = draw.textbbox((0, 0), text, font=font)
    return int(bbox[2] - bbox[0])


def _truncate(draw, text, font, max_width):
    text = text.rstrip()
    if _text_width(draw, text, font) <= max_width:
        return text
    while text and _text_width(draw, text + "…", font) > max_width:
        text = text[:-1]
    return (text + "…") if text else "…"


def _wrap(draw, text, font, max_width, max_lines):
    """Greedy word wrap with ellipsis on overflow."""
    if not text:
        return [""]
    words = text.split()
    lines: list[str] = []
    line = ""
    for word in words:
        candidate = f"{line} {word}".strip()
        if _text_width(draw, candidate, font) <= max_width:
            line = candidate
            continue
        if line:
            lines.append(line)
        if _text_width(draw, word, font) > max_width:
            lines.append(_truncate(draw, word, font, max_width))
            line = ""
        else:
            line = word
        if len(lines) >= max_lines:
            break
    if line and len(lines) < max_lines:
        lines.append(line)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
    rendered = " ".join(lines)
    if rendered.strip() != text.strip() and lines:
        lines[-1] = _truncate(draw, lines[-1], font, max_width)
    return lines


def render_captioned_grid(
    items: list[tuple[torch.Tensor | None, str]],
    *,
    ncols: int = 4,
    cell_w: int = 256,
    cell_h: int = 256,
    caption_h: int = 96,
    pad: int = 12,
    font_size: int = 14,
    max_caption_lines: int = 4,
    font_path: str | None = None,
    title: str | None = None,
    title_size: int = 22,
) -> Image.Image:
    """Render image-caption cells, suppressing the caption row when empty."""
    n = len(items)
    if n == 0:
        raise ValueError("render_captioned_grid received no items.")
    nrows = (n + ncols - 1) // ncols

    any_caption = any(c for _, c in items)
    cap_strip = caption_h if any_caption else 0
    inner_gap = pad if any_caption else 0

    cell_block_w = cell_w + 2 * pad
    cell_block_h = cell_h + inner_gap + cap_strip + 2 * pad
    title_strip = (title_size + 2 * pad) if title else 0
    W = ncols * cell_block_w + pad
    H = title_strip + nrows * cell_block_h + pad

    canvas = Image.new("RGB", (W, H), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    if title:
        title_font = _font(title_size, font_path)
        draw.text(
            (pad, pad),
            _truncate(draw, title, title_font, W - 2 * pad),
            fill=(0, 0, 0),
            font=title_font,
        )

    font = _font(font_size, font_path)
    line_h = font.getbbox("Hg")[3] - font.getbbox("Hg")[1] + 4

    for idx, (image_t, caption) in enumerate(items):
        r, c = divmod(idx, ncols)
        x0 = pad + c * cell_block_w
        y0 = title_strip + pad + r * cell_block_h

        # Image cell
        if image_t is not None:
            img = _tensor_to_pil(image_t).resize(
                (cell_w, cell_h), Image.Resampling.BICUBIC
            )
            canvas.paste(img, (x0 + pad, y0 + pad))
        else:
            draw.rectangle(
                (x0 + pad, y0 + pad, x0 + pad + cell_w, y0 + pad + cell_h),
                fill=(240, 240, 240),
                outline=(200, 200, 200),
            )

        if any_caption:
            cap_x = x0 + pad
            cap_y = y0 + pad + cell_h + inner_gap
            lines = _wrap(draw, caption or "", font, cell_w, max_caption_lines)
            for i, line in enumerate(lines):
                draw.text(
                    (cap_x, cap_y + i * line_h),
                    line,
                    fill=(20, 20, 20),
                    font=font,
                )

    return canvas


def save_wavs(
    wavs: Iterable[np.ndarray],
    out_dir: Path | str,
    *,
    sample_rate: int = 16000,
    prefix: str = "",
    digits: int = 2,
) -> list[Path]:
    import soundfile as sf

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for i, wav in enumerate(wavs):
        arr = wav if isinstance(wav, np.ndarray) else np.asarray(wav, dtype=np.float32)
        arr = arr.astype(np.float32, copy=False)
        if arr.ndim > 1:
            arr = arr.squeeze()
        path = out / f"{prefix}{i:0{digits}d}.wav"
        sf.write(str(path), arr, sample_rate)
        paths.append(path)
    return paths


def save_texts(
    texts: Iterable[str],
    path: Path | str,
    *,
    title: str | None = None,
) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        if title:
            f.write(f"# {title}\n\n")
        for i, t in enumerate(texts):
            f.write(f"[{i:02d}] {t}\n")
    return p


def save_translation(
    pair_dir,
    src,
    tgt,
    raw_src,
    raw_tgt,
    *,
    tag,
    ncols,
    sample_rate=16000,
):
    pair_dir.mkdir(parents=True, exist_ok=True)
    pair = frozenset({src, tgt})

    if pair == frozenset({"text", "image"}):
        # Image cell + caption-under-it in one PNG.
        if src == "text":
            items = [(raw_tgt[i], raw_src[i]) for i in range(len(raw_tgt))]
            title = f"{tag}: generated image ← source caption"
        else:
            items = [(raw_src[i], raw_tgt[i]) for i in range(len(raw_src))]
            title = f"{tag}: source image + generated caption"
        grid = render_captioned_grid(items, ncols=ncols, title=title)
        out = pair_dir / f"{tag}.png"
        grid.save(out)
        print(f"  wrote {out.name}")
        return

    if pair == frozenset({"text", "audio"}):
        if src == "text":
            save_texts(
                raw_src, pair_dir / "captions_src.txt", title=f"{tag}: source captions"
            )
            save_wavs(raw_tgt, pair_dir / "audio_gen", sample_rate=sample_rate)
        else:
            save_wavs(raw_src, pair_dir / "audio_src", sample_rate=sample_rate)
            save_texts(
                raw_tgt,
                pair_dir / "captions_gen.txt",
                title=f"{tag}: generated captions",
            )
        print(f"  wrote {pair_dir.name}/")
        return

    if pair == frozenset({"image", "audio"}):
        if src == "image":
            images = raw_src
            title = f"{tag}: source images (paired with audio_gen/XX.wav)"
            filename = "images_src.png"
        else:
            save_wavs(raw_src, pair_dir / "audio_src", sample_rate=sample_rate)
            images = raw_tgt
            title = f"{tag}: generated images (paired with audio_src/XX.wav)"
            filename = "images_gen.png"
        grid = render_captioned_grid([(image, "") for image in images], ncols=ncols, title=title)
        grid.save(pair_dir / filename)
        if src == "image":
            save_wavs(raw_tgt, pair_dir / "audio_gen", sample_rate=sample_rate)
        print(f"  wrote {pair_dir.name}/")
        return

    raise ValueError(f"Unknown modality pair: ({src}, {tgt})")


