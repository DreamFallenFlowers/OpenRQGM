"""Render the OpenRQGM architecture with directional flowing dash lines."""

from __future__ import annotations

import argparse
from io import BytesIO
from pathlib import Path

import cairosvg
from PIL import Image

SOURCE_WIDTH = 1600
SOURCE_HEIGHT = 980
DASH_PERIOD = 22
DASH_CYCLES_PER_LOOP = 4


def render_svg(svg_text: str, width: int, height: int, dash_offset: float) -> Image.Image:
    """Render one frame after advancing every flow line in path direction."""
    animated_rule = (
        ".cyan-line,.violet-line,.amber-line"
        f"{{stroke-dashoffset:{dash_offset:.3f};}}"
    )
    frame_svg = svg_text.replace("</style>", f"{animated_rule}</style>", 1)
    png = cairosvg.svg2png(
        bytestring=frame_svg.encode("utf-8"),
        output_width=width,
        output_height=height,
    )
    return Image.open(BytesIO(png)).convert("RGB")


def render(svg: Path, output: Path, width: int, frames: int, fps: int) -> None:
    height = round(width * SOURCE_HEIGHT / SOURCE_WIDTH)
    svg_text = svg.read_text(encoding="utf-8")

    # A negative SVG dash offset moves the pattern from each path's start toward
    # its marker-end. Ending on full periods keeps the GIF loop seamless.
    rgb_frames = [
        render_svg(
            svg_text,
            width,
            height,
            -DASH_PERIOD * DASH_CYCLES_PER_LOOP * index / frames,
        )
        for index in range(frames)
    ]

    palette = rgb_frames[0].quantize(colors=256, method=Image.Quantize.MEDIANCUT)
    rendered = [
        frame.quantize(palette=palette, dither=Image.Dither.NONE)
        for frame in rgb_frames
    ]

    output.parent.mkdir(parents=True, exist_ok=True)
    rendered[0].save(
        output,
        save_all=True,
        append_images=rendered[1:],
        duration=round(1000 / fps),
        loop=0,
        optimize=True,
        disposal=2,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--svg", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--frames", type=int, default=48)
    parser.add_argument("--fps", type=int, default=8)
    args = parser.parse_args()
    render(args.svg, args.output, args.width, args.frames, args.fps)


if __name__ == "__main__":
    main()
