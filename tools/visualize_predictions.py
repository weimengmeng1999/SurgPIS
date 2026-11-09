#!/usr/bin/env python3
"""
Colorize saved SurgPIS/Mask2Former label-map predictions (part-seg, instance,
or semantic id maps saved as flat-grey PNGs) and overlay them on their
original RGB frame.

These maps are saved by `pps_inference` / `demo_inspart` as single-channel
uint8 PNGs where each pixel value is a class or instance id -- they look
almost solid black when opened directly because the ids are small integers.

Usage:
    python tools/visualize_predictions.py \\
        --pred /path/to/pred1.png /path/to/pred_dir \\
        --images-root /path/to/EndoVis-RS17/val/images /path/to/other/images \\
        --output tools/vis_output

For each prediction file, the script searches --images-root (recursively)
for a source image with the same basename, then writes a single PNG with
three panels: original | colorized labels | overlay.
"""
import argparse
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Mirrors mask2former/data/datasets/register_endovis_inp.py::ROBO_CATEGORIES.
# INSTANCE_PART_CATEGORIES_INF assigns each instrument 3 consecutive part ids
# (id 0 is background): category k (1-indexed) owns ids [3k-2, 3k-1, 3k].
ROBO_CATEGORIES = [
    "Bipolar Forceps", "Prograsp Forceps", "Large Needle Driver",
    "Monopolar Curved Scissors", "Ultrasound Probe", "Suction Instrument",
    "Clip Applier", "Vessel Sealer", "Grasping Retractor",
]
PART_NAMES = ["Shaft", "Wrist", "Clasper"]

# A fixed, high-contrast qualitative palette (Sasha Trubetskoy's 20-color set)
# so the same label id always maps to the same color across runs/frames.
PALETTE_HEX = [
    "#e6194B", "#3cb44b", "#ffe119", "#4363d8", "#f58231", "#911eb4", "#42d4f4",
    "#f032e6", "#bfef45", "#fabed4", "#469990", "#dcbeff", "#9A6324", "#fffac8",
    "#800000", "#aaffc3", "#808000", "#ffd8b1", "#000075", "#a9a9a9",
]


def hex_to_rgb(h):
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


PALETTE = [hex_to_rgb(h) for h in PALETTE_HEX]


def label_name(label_id, scheme):
    if label_id == 0:
        return "background"
    if scheme == "part":
        idx = label_id - 1
        cat, part = idx // 3, idx % 3
        instrument = ROBO_CATEGORIES[cat] if cat < len(ROBO_CATEGORIES) else f"category {cat}"
        part_name = PART_NAMES[part] if part < len(PART_NAMES) else f"part {part}"
        return f"{instrument} · {part_name}"
    if scheme == "instance":
        # ins_map ids are just a per-image sequential counter (see
        # pps_inference in maskformer_model.py) -- no class/part semantics,
        # so there's nothing to decode beyond "which instance is this".
        return f"instance {label_id}"
    return f"label {label_id}"


def colorize(label_map):
    """label_map: HxW uint8/int array. Returns HxW x3 uint8 RGB, background (id 0) black."""
    h, w = label_map.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for lid in np.unique(label_map):
        if lid == 0:
            continue
        color = PALETTE[(int(lid) - 1) % len(PALETTE)]
        rgb[label_map == lid] = color
    return rgb


def find_source_image(basename, images_roots):
    for root in images_roots:
        for dirpath, _, filenames in os.walk(root):
            if basename in filenames:
                return os.path.join(dirpath, basename)
    return None


def guess_scheme(label_map):
    # Part-id maps top out at 27 (9 instruments x 3 parts); treat anything
    # in that range as part ids, otherwise just label them generically.
    return "part" if label_map.max() <= 27 else "generic"


def resolve_scheme(pred_path, label_map, mode):
    """mode: 'auto', 'part', or 'instance'. 'auto' looks at the containing
    directory name (endo17/ins vs endo17/inspart, matching where
    pps_inference in maskformer_model.py saves each map) before falling back
    to the old value-range guess -- ins_map ids collide numerically with
    small part-category ids, so the directory is the only reliable signal.
    """
    if mode != "auto":
        return mode
    parent = os.path.basename(os.path.dirname(os.path.abspath(pred_path)))
    if parent == "ins":
        return "instance"
    if parent == "inspart":
        return "part"
    return guess_scheme(label_map)


def make_legend(label_map, scheme, swatch=14, pad=8, font=None):
    ids = [i for i in np.unique(label_map) if i != 0]
    if not ids:
        return None
    line_h = swatch + 6
    width = 260
    img = Image.new("RGB", (width, pad * 2 + line_h * len(ids)), (18, 18, 18))
    draw = ImageDraw.Draw(img)
    for i, lid in enumerate(ids):
        y = pad + i * line_h
        color = PALETTE[(int(lid) - 1) % len(PALETTE)]
        draw.rectangle([pad, y, pad + swatch, y + swatch], fill=color)
        draw.text((pad + swatch + 8, y - 1), f"{lid}  {label_name(int(lid), scheme)}",
                   fill=(235, 235, 235), font=font)
    return img


def build_panel(image_path, pred_path, alpha, mode="auto"):
    label_map = np.array(Image.open(pred_path))
    if label_map.ndim == 3:
        label_map = label_map[..., 0]
    scheme = resolve_scheme(pred_path, label_map, mode)
    # ins_map (unlike inspart) encodes background as 255, not 0.
    if scheme == "instance" and (label_map == 255).any():
        label_map = np.where(label_map == 255, 0, label_map)

    if image_path is not None:
        original = np.array(Image.open(image_path).convert("RGB").resize(
            (label_map.shape[1], label_map.shape[0])))
    else:
        original = np.zeros((*label_map.shape, 3), dtype=np.uint8)

    colored = colorize(label_map)
    mask = label_map != 0
    overlay = original.copy()
    overlay[mask] = (original[mask] * (1 - alpha) + colored[mask] * alpha).astype(np.uint8)

    h, w = label_map.shape
    gap = 6
    canvas = Image.new("RGB", (w * 3 + gap * 2, h), (10, 10, 10))
    canvas.paste(Image.fromarray(original), (0, 0))
    canvas.paste(Image.fromarray(colored), (w + gap, 0))
    canvas.paste(Image.fromarray(overlay), (2 * (w + gap), 0))

    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.load_default(size=22)
    except TypeError:
        font = ImageFont.load_default()
    for i, title in enumerate(["original", "colorized prediction", "overlay"]):
        x = i * (w + gap) + 12
        draw.text((x + 1, 11), title, fill=(0, 0, 0), font=font)
        draw.text((x, 10), title, fill=(255, 255, 255), font=font)

    legend = make_legend(label_map, scheme, font=font)
    if legend is not None:
        combined = Image.new("RGB", (canvas.width, canvas.height + legend.height), (18, 18, 18))
        combined.paste(canvas, (0, 0))
        combined.paste(legend, (0, canvas.height))
        canvas = combined

    return canvas


def collect_pred_files(pred_args):
    files = []
    for p in pred_args:
        if os.path.isdir(p):
            files += [os.path.join(p, f) for f in sorted(os.listdir(p)) if f.lower().endswith(".png")]
        else:
            files.append(p)
    return files


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pred", nargs="+", required=True,
                     help="prediction PNG file(s) and/or directories of PNGs")
    ap.add_argument("--images-root", nargs="+", required=True,
                     help="directory (or directories) to search for the matching original image")
    ap.add_argument("--output", default="tools/vis_output",
                     help="output directory (default: tools/vis_output)")
    ap.add_argument("--alpha", type=float, default=0.55, help="overlay blend strength (0-1)")
    ap.add_argument("--mode", choices=["auto", "part", "instance"], default="auto",
                     help="how to interpret label ids: 'part' decodes them as "
                          "instrument/part categories (endo17/inspart maps), "
                          "'instance' treats them as plain per-image instance "
                          "ids with no class semantics (endo17/ins maps, which "
                          "also use 255 rather than 0 for background). 'auto' "
                          "(default) infers this from the containing directory "
                          "name, falling back to a value-range guess.")
    args = ap.parse_args()

    os.makedirs(args.output, exist_ok=True)
    pred_files = collect_pred_files(args.pred)
    if not pred_files:
        print("No prediction PNGs found.", file=sys.stderr)
        sys.exit(1)

    for pred_path in pred_files:
        basename = os.path.basename(pred_path)
        image_path = find_source_image(basename, args.images_root)
        if image_path is None:
            print(f"[warn] no source image named '{basename}' found under {args.images_root}; "
                  f"visualizing prediction only (no background).", file=sys.stderr)
        else:
            print(f"{basename}: matched {image_path}")

        canvas = build_panel(image_path, pred_path, args.alpha, args.mode)
        out_path = os.path.join(args.output, basename)
        canvas.save(out_path)
        print(f"  -> {out_path}")


if __name__ == "__main__":
    main()
