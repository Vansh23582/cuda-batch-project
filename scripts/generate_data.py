#!/usr/bin/env python3
"""
scripts/generate_data.py

I wrote this to create synthetic grayscale PGM images for testing the
CUDA batch processor without needing to download a real dataset.

It generates `--count` images of size `--size x --size` pixels, each with
a random mix of shapes (circles, rectangles, Gaussian blobs) so that the
Sobel edge detector has something interesting to find.

Usage:
    python3 scripts/generate_data.py --count 120 --size 256 --outdir data

Requirements:
    pip install Pillow numpy
"""

import argparse
import os
import random
import struct
import numpy as np

try:
    from PIL import Image, ImageDraw
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


def write_pgm(path: str, pixels: np.ndarray) -> None:
    """
    Write a numpy uint8 2D array as a binary PGM (P5) file.
    I implement this manually so the script works even without Pillow
    (Pillow's save() works too, but having a fallback is nice).
    """
    H, W = pixels.shape
    with open(path, "wb") as f:
        # PGM header: magic, width, height, maxval
        header = f"P5\n{W} {H}\n255\n"
        f.write(header.encode("ascii"))
        f.write(pixels.astype(np.uint8).tobytes())


def generate_image_numpy(size: int, seed: int) -> np.ndarray:
    """
    Generate one synthetic grayscale image using only NumPy.
    This is the fallback when Pillow is not installed.

    Strategy: start with a noisy dark background, then stamp random
    bright rectangles so Sobel has edges to detect.
    """
    rng = np.random.default_rng(seed)
    # Low-level noise background (makes histogram equalization meaningful)
    img = rng.integers(20, 60, size=(size, size), dtype=np.uint8)

    # Add 5–15 bright rectangles
    n_rects = rng.integers(5, 15)
    for _ in range(n_rects):
        x0 = int(rng.integers(0, size - 1))
        y0 = int(rng.integers(0, size - 1))
        x1 = int(min(x0 + rng.integers(10, size // 3), size - 1))
        y1 = int(min(y0 + rng.integers(10, size // 3), size - 1))
        brightness = int(rng.integers(150, 255))
        img[y0:y1, x0:x1] = brightness

    return img


def generate_image_pillow(size: int, seed: int) -> np.ndarray:
    """
    Generate one synthetic grayscale image using Pillow + NumPy.
    Produces circles, rectangles, and Gaussian blobs for variety.
    """
    rng = np.random.default_rng(seed)

    # Noisy background (dark, so histogram equalization has room to work)
    base = rng.integers(10, 50, size=(size, size), dtype=np.uint8).astype(np.float32)
    img_pil = Image.fromarray(base.astype(np.uint8), mode="L")
    draw = ImageDraw.Draw(img_pil)

    # Random rectangles
    for _ in range(rng.integers(3, 8)):
        x0, y0 = int(rng.integers(0, size)), int(rng.integers(0, size))
        x1, y1 = int(rng.integers(x0, min(x0 + size // 3, size))), \
                  int(rng.integers(y0, min(y0 + size // 3, size)))
        fill = int(rng.integers(140, 255))
        draw.rectangle([x0, y0, x1, y1], fill=fill)

    # Random ellipses (gives curved edges for Sobel to find)
    for _ in range(rng.integers(2, 6)):
        cx, cy = int(rng.integers(0, size)), int(rng.integers(0, size))
        rx, ry = int(rng.integers(5, size // 5)), int(rng.integers(5, size // 5))
        fill = int(rng.integers(100, 220))
        draw.ellipse([cx - rx, cy - ry, cx + rx, cy + ry], fill=fill)

    # Add a Gaussian blob (soft bright spot) via NumPy then paste back
    arr = np.array(img_pil, dtype=np.float32)
    bx = int(rng.integers(size // 4, 3 * size // 4))
    by = int(rng.integers(size // 4, 3 * size // 4))
    sigma = rng.integers(size // 10, size // 4)
    Y, X = np.ogrid[:size, :size]
    blob = np.exp(-((X - bx) ** 2 + (Y - by) ** 2) / (2 * sigma ** 2))
    strength = rng.integers(80, 180)
    arr = np.clip(arr + blob * strength, 0, 255).astype(np.uint8)

    return arr


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic PGM images for CUDA batch processing test.")
    parser.add_argument("--count",  type=int, default=120,
                        help="Number of images to generate (default: 120)")
    parser.add_argument("--size",   type=int, default=256,
                        help="Image width and height in pixels (default: 256)")
    parser.add_argument("--outdir", type=str, default="data",
                        help="Output directory (default: data/)")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    gen_fn = generate_image_pillow if HAS_PIL else generate_image_numpy
    mode   = "Pillow" if HAS_PIL else "NumPy-only"
    print(f"Generating {args.count} images ({args.size}x{args.size} px) "
          f"using {mode} → '{args.outdir}/'")

    for i in range(args.count):
        pixels = gen_fn(args.size, seed=i)
        fname  = os.path.join(args.outdir, f"synthetic_{i:04d}.pgm")
        write_pgm(fname, pixels)

        # Progress every 20 images
        if (i + 1) % 20 == 0 or (i + 1) == args.count:
            print(f"  {i + 1}/{args.count} done")

    print(f"\nAll {args.count} images written to '{args.outdir}/'")
    print("Now run:  make all && ./run.sh")


if __name__ == "__main__":
    main()
