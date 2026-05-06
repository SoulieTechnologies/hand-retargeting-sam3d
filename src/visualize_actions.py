"""
Plot joint-angle trajectories saved by kinematics.py (Pinocchio configuration vector q).

Usage (from repo root, after generating actions):
  python src/visualize_actions.py
  python src/visualize_actions.py --actions data/actions/orcahand_actions.npy --save out/actions_plot.png
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from datetime import datetime

import numpy as np

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))


def _default_actions_path() -> str:
    return os.path.join(_PROJECT_ROOT, "data", "actions", "orcahand_actions.npy")


def _default_urdf_path() -> str:
    return os.path.join(_PROJECT_ROOT, "assets", "orcahand", "orcahand_scaled.urdf")


def _fingerprint(a: np.ndarray) -> str:
    """Short hash so you can tell if the .npy changed without re-reading the image."""
    h = hashlib.md5()
    h.update(str(a.shape).encode())
    h.update(np.ascontiguousarray(a, dtype=np.float64).ravel()[: min(a.size, 4096)].tobytes())
    h.update(np.ascontiguousarray(a, dtype=np.float64).ravel()[-min(a.size, 4096) :].tobytes())
    h.update(f"{float(a.mean()):.12g}{float(a.std()):.12g}".encode())
    return h.hexdigest()[:12]


def load_joint_labels(urdf_path: str) -> list[str]:
    try:
        import pinocchio as pin

        model = pin.buildModelFromUrdf(urdf_path)
    except Exception as e:
        print(
            f"Warning: could not load URDF for joint names ({e}); using generic labels.",
            file=sys.stderr,
        )
        return []

    labels: list[str] = []
    for jid in range(1, model.njoints):
        j = model.joints[jid]
        if j.nq == 0:
            continue
        name = model.names[jid]
        for k in range(j.nq):
            if j.nq > 1:
                labels.append(f"{name}[{k}]")
            else:
                labels.append(name)
    return labels


def main() -> None:
    p = argparse.ArgumentParser(description="Plot IK joint trajectories (orcahand_actions.npy).")
    p.add_argument(
        "--actions",
        type=str,
        default=_default_actions_path(),
        help="Path to .npy with shape (T, nq) from kinematics.py",
    )
    p.add_argument(
        "--urdf",
        type=str,
        default=_default_urdf_path(),
        help="URDF used in kinematics.py (for joint names and limits)",
    )
    p.add_argument("--fps", type=float, default=30.0, help="FPS for time axis (only affects x labels)")
    p.add_argument(
        "--save",
        type=str,
        default="",
        help="If set, save figure to this path instead of showing interactively",
    )
    p.add_argument("--dpi", type=int, default=120)
    args = p.parse_args()

    actions_path = os.path.abspath(args.actions)
    if not os.path.isfile(actions_path):
        print(f"Error: actions file not found: {actions_path}", file=sys.stderr)
        sys.exit(1)

    q = np.load(actions_path)
    if q.ndim != 2:
        print(f"Error: expected 2D array (T, nq), got shape {q.shape}", file=sys.stderr)
        sys.exit(1)

    num_frames, nq = q.shape
    mtime = os.path.getmtime(actions_path)
    mtime_s = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
    fp = _fingerprint(q)
    print(
        f"Loaded actions:\n"
        f"  path:       {actions_path}\n"
        f"  modified:   {mtime_s}\n"
        f"  shape:      {q.shape}\n"
        f"  fingerprint {fp}\n"
        f"If the plot never changes, regenerate data/actions first: python src/kinematics.py"
    )
    labels = load_joint_labels(os.path.abspath(args.urdf))
    if len(labels) != nq:
        if labels:
            print(
                f"Warning: URDF reports {len(labels)} scalar DoFs but actions have nq={nq}; "
                "using generic labels.",
                file=sys.stderr,
            )
        labels = [f"q[{i}]" for i in range(nq)]

    limits = None
    try:
        import pinocchio as pin

        model = pin.buildModelFromUrdf(os.path.abspath(args.urdf))
        if model.nq == nq:
            limits = (model.lowerPositionLimit.copy(), model.upperPositionLimit.copy())
    except Exception:
        pass

    import matplotlib.pyplot as plt

    n = nq
    ncols = min(3, max(1, n))
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 2.2 * nrows), sharex=True, squeeze=False)
    axes_flat = axes.ravel()

    if args.fps > 0:
        x = np.arange(num_frames) / float(args.fps)
        xlabel = "Time (s)"
    else:
        x = np.arange(num_frames)
        xlabel = "Frame index"

    for i in range(n):
        ax = axes_flat[i]
        ax.plot(x, q[:, i], color="C0", lw=1.0)
        ax.set_ylabel(labels[i], fontsize=8, rotation=0, ha="right", va="center")
        ax.tick_params(axis="y", labelsize=7)
        if limits is not None:
            lo, hi = float(limits[0][i]), float(limits[1][i])
            ax.axhline(lo, color="gray", ls="--", lw=0.6, alpha=0.7)
            ax.axhline(hi, color="gray", ls="--", lw=0.6, alpha=0.7)
            ax.set_ylim(lo - 0.05 * (hi - lo + 1e-6), hi + 0.05 * (hi - lo + 1e-6))

    for j in range(n, len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.suptitle(
        f"Joint angles — {num_frames} frames, nq={nq}\n"
        f"{os.path.basename(actions_path)} · mtime {mtime_s} · fp {fp}",
        fontsize=10,
    )
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    fig.supxlabel(xlabel, fontsize=9)
    if args.save:
        out = os.path.abspath(args.save)
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
        print(f"Saved figure to {out}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
