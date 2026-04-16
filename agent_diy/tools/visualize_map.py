"""
visualize_map.py
Read all .npy map files under MAP_DIR and export same-name images to OUT_DIR.

Usage:
    python agent_diy/tools/visualize_map.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

MAP_DIR = Path(__file__).resolve().parents[1] / "monitor" / "eval_snapshots"
OUT_DIR = Path(__file__).resolve().parent / "output"


def render_npy_map(npy_path: Path, out_dir: Path) -> None:
    arr = np.load(npy_path)
    if arr.ndim != 2:
        raise ValueError(f"expected 2D array, got shape={arr.shape} from {npy_path.name}")

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(arr, cmap="viridis", origin="upper", interpolation="nearest")
    ax.set_title(npy_path.stem)
    ax.axis("off")

    out_path = out_dir / f"{npy_path.stem}.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved -> {out_path}")


def main() -> None:
    npy_files = sorted(MAP_DIR.glob("*.npy"))
    if not npy_files:
        print(f"[warn] no npy files found under: {MAP_DIR}")
        return

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"input dir:  {MAP_DIR}")
    print(f"output dir: {OUT_DIR}")

    for npy_path in npy_files:
        render_npy_map(npy_path, OUT_DIR)


if __name__ == "__main__":
    main()
