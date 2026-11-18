"""Decompress FACEIT demos (.dem.zst / .dem.gz) in data/raw/demos into plain .dem files.

Usage:
    python src/pipeline/prepare_demos.py           # decompress new ones, keep the originals
    python src/pipeline/prepare_demos.py --delete  # also delete the compressed file after
"""

from __future__ import annotations

import argparse
import gzip
import shutil
from pathlib import Path

import zstandard

DEMOS = Path(__file__).resolve().parents[2] / "data" / "raw" / "demos"


def decompress(src: Path) -> Path:
    dest = src.with_suffix("")  # x.dem.zst -> x.dem
    if src.suffix == ".zst":
        with src.open("rb") as fin, dest.open("wb") as fout:
            zstandard.ZstdDecompressor().copy_stream(fin, fout)
    elif src.suffix == ".gz":
        with gzip.open(src, "rb") as fin, dest.open("wb") as fout:
            shutil.copyfileobj(fin, fout)
    return dest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--delete", action="store_true", help="delete compressed files after")
    args = parser.parse_args()

    packed = sorted(p for p in DEMOS.iterdir() if p.suffix in {".zst", ".gz"})
    if not packed:
        print(f"No compressed demos in {DEMOS}")
    for src in packed:
        dest = src.with_suffix("")
        if dest.exists():
            print(f"skip   {src.name} (already decompressed)")
        else:
            decompress(src)
            print(f"done   {dest.name} ({dest.stat().st_size / 1e6:.0f} MB)")
        if args.delete:
            src.unlink()


if __name__ == "__main__":
    main()
