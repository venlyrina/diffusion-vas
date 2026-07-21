import argparse
from pathlib import Path
import numpy as np
import torch


def vggt_depth_to_diffvas(
    npz_path,
    depth_key="depth",
    expected_frames=None,
    invert=True,
    normalization="minmax",
    percentile_low=1.0,
    percentile_high=99.0,
):
    """
    Convert VGGT depth to Diffusion-VAS amodal depth conditioning.

    Input VGGT depth:
        [F,H,W,1] or [F,H,W]

    Output:
        torch.float32 [1,F,3,H,W], range [-1,1]

    Default minmax normalization matches demo.py's DA2 preprocessing.
    """
    path = Path(npz_path)
    if not path.is_file():
        raise FileNotFoundError(f"VGGT NPZ not found: {path}")

    with np.load(path, allow_pickle=False) as data:
        if depth_key not in data.files:
            raise KeyError(
                f"Key {depth_key!r} not found. Available keys: {data.files}"
            )
        depth = np.asarray(data[depth_key], dtype=np.float32)

    if depth.ndim == 4 and depth.shape[-1] == 1:
        depth = depth[..., 0]

    if depth.ndim != 3:
        raise ValueError(
            f"Expected [F,H,W] or [F,H,W,1], got {depth.shape}"
        )

    if expected_frames is not None and depth.shape[0] != expected_frames:
        raise ValueError(
            f"Frame mismatch: VGGT={depth.shape[0]}, "
            f"Diff-VAS={expected_frames}. Refusing silent truncate/repeat."
        )

    finite = np.isfinite(depth)
    if not finite.all():
        if not finite.any():
            raise ValueError("Depth contains no finite values.")
        depth = np.where(depth == depth, depth, np.median(depth[finite]))
        depth = np.nan_to_num(
            depth,
            nan=float(np.median(depth[finite])),
            posinf=float(depth[finite].max()),
            neginf=float(depth[finite].min()),
        ).astype(np.float32)

    raw_min, raw_max = float(depth.min()), float(depth.max())

    if normalization == "minmax":
        lo, hi = raw_min, raw_max
    elif normalization == "percentile":
        lo, hi = np.percentile(
            depth, [percentile_low, percentile_high]
        )
        lo, hi = float(lo), float(hi)
    else:
        raise ValueError("normalization must be 'minmax' or 'percentile'")

    if hi - lo <= 1e-8:
        raise ValueError(f"Degenerate depth range: {lo}..{hi}")

    # Sequence-global normalization, matching Diff-VAS demo.py.
    depth = np.clip((depth - lo) / (hi - lo), 0.0, 1.0)

    # Current compatibility hypothesis from VGGT-vs-DA2 audit.
    if invert:
        depth = 1.0 - depth

    # Match DA2 path in demo.py.
    depth = depth * 2.0 - 1.0

    # [F,H,W] -> [1,F,3,H,W]
    tensor = torch.from_numpy(
        np.ascontiguousarray(depth.astype(np.float32))
    )
    tensor = tensor.unsqueeze(1).repeat(1, 3, 1, 1).unsqueeze(0)

    print(
        f"[VGGT->DiffVAS] raw={raw_min:.6f}..{raw_max:.6f} "
        f"output_shape={tuple(tensor.shape)} "
        f"output_range={tensor.min().item():.6f}..{tensor.max().item():.6f} "
        f"invert={invert} normalization={normalization}"
    )
    return tensor


def main():
    p = argparse.ArgumentParser()
    p.add_argument("vggt_npz")
    p.add_argument("--depth-key", default="depth")
    p.add_argument("--expected-frames", type=int)
    p.add_argument("--no-invert", action="store_true")
    p.add_argument(
        "--normalization",
        choices=["minmax", "percentile"],
        default="minmax",
    )
    p.add_argument("--save")
    args = p.parse_args()

    tensor = vggt_depth_to_diffvas(
        args.vggt_npz,
        depth_key=args.depth_key,
        expected_frames=args.expected_frames,
        invert=not args.no_invert,
        normalization=args.normalization,
    )

    if args.save:
        torch.save(tensor, args.save)
        print(f"Saved: {args.save}")


if __name__ == "__main__":
    main()
