"""
visualize_map_downsample.py
Compare multiple downsampling resolutions (128 -> 64 / 32 / 16) for all maps,
using both mean-pooling and nearest-neighbor strategies.

Usage:
    python agent_diy/tools/visualize_map_downsample.py
"""

import json
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

MAP_SIZE = 128
CANDIDATE_SIZES = [64, 32, 16]   # output resolutions to compare

MAP_DIR = Path(__file__).resolve().parents[1] / "ref" / "map"
OUT_DIR = Path(__file__).resolve().parent / "output"


# ─────────────────────────── helpers ────────────────────────────────────────

def load_map(path: Path) -> np.ndarray:
    """Parse JSON map -> 128x128 binary walkability matrix (1=walk, 0=wall)."""
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    matrix = np.zeros((MAP_SIZE, MAP_SIZE), dtype=np.float32)
    for cell in data.get("cells", []):
        x, z = int(cell["x"]), int(cell["z"])
        if 0 <= x < MAP_SIZE and 0 <= z < MAP_SIZE and int(cell.get("type_id", 0)) == 1:
            matrix[z, x] = 1.0
    return matrix


def downsample_mean(arr: np.ndarray, target: int) -> np.ndarray:
    factor = MAP_SIZE // target
    return arr.reshape(target, factor, target, factor).mean(axis=(1, 3))


def downsample_nearest(arr: np.ndarray, target: int) -> np.ndarray:
    """Pick center pixel of each block (nearest-neighbor)."""
    factor = MAP_SIZE // target
    offset = factor // 2
    return arr[offset::factor, offset::factor][:target, :target]


def mixed_block_ratio(ds: np.ndarray) -> float:
    """Fraction of blocks that contain both walkable and wall cells (mean encoding)."""
    total = ds.size
    mixed = int(((ds > 0.25) & (ds < 1.0)).sum())
    return mixed / total


def corridor_preservation(raw: np.ndarray, ds: np.ndarray, target: int) -> float:
    """
    Estimate what fraction of walkable-corridor pixels survive into the
    downsampled map as 'pure walkable' blocks (value == 1.0 for mean,
    value == 1.0 for nearest).
    """
    factor = MAP_SIZE // target
    walkable_blocks_raw = (
        raw.reshape(target, factor, target, factor).min(axis=(1, 3))
    )  # 1.0 only if ALL pixels in block are walkable
    preserved = float(walkable_blocks_raw.sum()) * factor * factor
    total_walkable = float(raw.sum())
    return preserved / total_walkable if total_walkable > 0 else 0.0


# ─────────────────────────── per-map visualisation ──────────────────────────

def visualize_map(map_path: Path, raw: np.ndarray, out_dir: Path) -> None:
    """
    One figure per map:
      row 0 : mean downsample  for each candidate size (+ original)
      row 1 : nearest downsample for each candidate size
    """
    map_name = map_path.stem
    n_sizes = len(CANDIDATE_SIZES)
    fig, axes = plt.subplots(2, n_sizes + 1, figsize=(4 * (n_sizes + 1), 8))
    fig.suptitle(map_name, fontsize=13)

    # original (shown twice, top-left and bottom-left)
    for row in range(2):
        axes[row, 0].imshow(raw, cmap="gray", origin="upper",
                            vmin=0, vmax=1, interpolation="nearest")
        axes[row, 0].set_title("Original 128x128")
        axes[row, 0].axis("off")

    axes[0, 0].set_ylabel("mean-pool", fontsize=10)
    axes[1, 0].set_ylabel("nearest-neighbor", fontsize=10)

    for col, sz in enumerate(CANDIDATE_SIZES, start=1):
        ds_mean = downsample_mean(raw, sz)
        ds_nn   = downsample_nearest(raw, sz)

        for row, ds in enumerate([ds_mean, ds_nn]):
            method = "mean" if row == 0 else "nearest"
            im = axes[row, col].imshow(ds, cmap="gray", origin="upper",
                                       vmin=0, vmax=1, interpolation="nearest")
            mr = mixed_block_ratio(ds) if row == 0 else float(((ds != 0.0) & (ds != 1.0)).mean())
            axes[row, col].set_title(
                f"{method} {sz}x{sz}\nmixed={mr:.1%}"
            )
            axes[row, col].axis("off")
            # grid lines
            for i in range(sz + 1):
                axes[row, col].axhline(i - 0.5, color="gray", linewidth=0.2, alpha=0.5)
                axes[row, col].axvline(i - 0.5, color="gray", linewidth=0.2, alpha=0.5)

    plt.tight_layout()
    out_path = out_dir / f"{map_name}_compare.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> {out_path}")


# ─────────────────────────── summary table ──────────────────────────────────

def main():
    map_files = sorted(MAP_DIR.glob("gorge_chase_map_*.json"))
    if not map_files:
        print(f"[error] no map files found, check: {MAP_DIR}", file=sys.stderr)
        sys.exit(1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"output dir: {OUT_DIR}\n")

    # header
    size_headers = "".join(
        f"  mean{sz}(mix%)  nn{sz}(mix%)" for sz in CANDIDATE_SIZES
    )
    print(f"{'map':<26}{size_headers}")
    print("-" * (26 + len(size_headers)))

    for path in map_files:
        raw = load_map(path)
        row_str = f"{path.stem:<26}"
        for sz in CANDIDATE_SIZES:
            ds_mean = downsample_mean(raw, sz)
            ds_nn   = downsample_nearest(raw, sz)
            mr_mean = mixed_block_ratio(ds_mean)
            # nearest: mixed = non-binary pixels (should be 0 for binary input)
            mr_nn = 0.0  # nearest on binary input is always binary
            row_str += f"  {mr_mean:>12.1%}  {mr_nn:>10.1%}"
        print(row_str)
        print(f"  processing: {path.name}")
        visualize_map(path, raw, OUT_DIR)

    print()
    print("Interpretation:")
    print("  mixed%  : fraction of blocks that blend both walkable and wall pixels")
    print("  nearest : always 0% mixed (picks one real pixel per block)")
    print()
    print("Recommendation:")
    print("  64x64 mean  : best detail preservation, moderate cost")
    print("  64x64 nearest: zero blurring, cleanest corridors, same cost")
    print("  32x32 nearest: half the size, still reasonable for coarse navigation")


if __name__ == "__main__":
    main()
