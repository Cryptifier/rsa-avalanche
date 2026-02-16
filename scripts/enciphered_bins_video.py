#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path
import sys
from typing import Optional

import matplotlib

matplotlib.use("Agg")
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from multiprocessing import Pool, cpu_count
import subprocess
import shutil


def load_bins(path: Path, metric: str) -> np.ndarray:
    rows = []
    max_frame = -1
    max_bin = -1

    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if not row:
                continue
            head = row[0].strip()
            if head.startswith("#") or head == "frame_index":
                continue
            try:
                frame_idx = int(row[0])
                bin_idx = int(row[3])
                value = float(row[5]) if metric == "float" else float(row[4])
            except (ValueError, IndexError):
                continue
            rows.append((frame_idx, bin_idx, value))
            max_frame = max(max_frame, frame_idx)
            max_bin = max(max_bin, bin_idx)

    if max_frame < 0 or max_bin < 0:
        raise RuntimeError("no data rows found in CSV")

    data = np.zeros((max_frame + 1, max_bin + 1), dtype=float)
    for frame_idx, bin_idx, value in rows:
        data[frame_idx, bin_idx] = value
    if metric == "float":
        max_val = float(np.max(data)) if data.size else 0.0
        if max_val <= 1.5:
            data *= 100.0
    return data


def render_video(
    data: np.ndarray,
    output: Path,
    fps: int,
    dpi: int,
    window: int,
    frame_step: int,
    max_frames: Optional[int],
    elev: float,
    azim: float,
    azim_step: float,
    metric: str,
    cmap: str,
    z_percentile_low: float,
    z_percentile_high: float,
    z_min: Optional[float],
    z_max: Optional[float],
    z_scale: str,
    smooth_window: int,
) -> None:
    data_plot = smooth_data(data, smooth_window)
    frames_total, bin_count = data_plot.shape
    frame_indices = list(range(0, frames_total, max(1, frame_step)))
    if max_frames is not None:
        frame_indices = frame_indices[: max_frames]

    fig = plt.figure(figsize=(11, 7))
    ax = fig.add_subplot(111, projection="3d")
    x = np.arange(bin_count)
    z_flat = data_plot.flatten() if data_plot.size else np.array([0.0])
    z_flat_scaled = apply_scale(z_flat, z_scale)
    if z_min is not None and z_max is not None:
        z_low = min(z_min, z_max)
        z_high = max(z_min, z_max)
    else:
        z_percentile_low = max(0.0, min(100.0, z_percentile_low))
        z_percentile_high = max(z_percentile_low, min(100.0, z_percentile_high))
        z_low = float(np.percentile(z_flat_scaled, z_percentile_low))
        z_high = float(np.percentile(z_flat_scaled, z_percentile_high))
    if z_high <= z_low:
        z_low = float(np.min(z_flat_scaled))
        z_high = float(np.max(z_flat_scaled))
    if z_high <= z_low:
        z_high = z_low + 1.0
    z_label = "match_pct" if metric == "float" else "match_count"

    def draw(frame_pos: int):
        frame_idx = frame_indices[frame_pos]
        start = max(0, frame_idx - window + 1)
        y = np.arange(start, frame_idx + 1)
        x_grid, y_grid = np.meshgrid(x, y)
        z_raw = data_plot[start : frame_idx + 1, :]
        z = apply_scale(z_raw, z_scale)

        ax.clear()
        ax.plot_surface(
            x_grid,
            y_grid,
            z,
            cmap=cmap,
            linewidth=0,
            antialiased=False,
        )
        flat = z_raw.ravel()
        if flat.size:
            top_n = min(3, flat.size)
            top_idx = np.argpartition(flat, -top_n)[-top_n:]
            y_idx, x_idx = np.unravel_index(top_idx, z_raw.shape)
            peak_vals = apply_scale(z_raw[y_idx, x_idx], z_scale)
            ax.scatter(
                x[x_idx],
                y[y_idx],
                peak_vals,
                c="red",
                s=45,
                depthshade=False,
            )
        ax.set_title(f"Enciphered bins (frame {frame_idx})")
        ax.set_xlabel("bin_index")
        ax.set_ylabel("frame_index")
        ax.set_zlabel(z_label)
        ax.set_xlim(0, bin_count - 1)
        ax.set_ylim(start, max(frame_idx, start + 1))
        ax.set_zlim(z_low, z_high)
        ax.view_init(elev=elev, azim=azim + frame_pos * azim_step)
        return fig,

    anim = animation.FuncAnimation(
        fig,
        draw,
        frames=len(frame_indices),
        interval=1000 / max(1, fps),
        blit=False,
    )

    try:
        writer = animation.FFMpegWriter(fps=fps)
        anim.save(output, writer=writer, dpi=dpi)
        print(f"Wrote video to {output}")
    except Exception as exc:
        frames_dir = output.with_suffix("")
        frames_dir = Path(f"{frames_dir}_frames")
        frames_dir.mkdir(parents=True, exist_ok=True)
        for idx in range(len(frame_indices)):
            draw(idx)
            frame_path = frames_dir / f"frame_{idx:04d}.png"
            fig.savefig(frame_path, dpi=dpi)
        print(f"FFmpeg unavailable ({exc}); wrote frames to {frames_dir}")


_WORKER_STATE = {}


def _init_worker(
    data: np.ndarray,
    x: np.ndarray,
    z_low: float,
    z_high: float,
    window: int,
    elev: float,
    azim: float,
    azim_step: float,
    cmap: str,
    z_scale: str,
    out_dir: Path,
    dpi: int,
    z_label: str,
) -> None:
    _WORKER_STATE["data"] = data
    _WORKER_STATE["x"] = x
    _WORKER_STATE["z_low"] = z_low
    _WORKER_STATE["z_high"] = z_high
    _WORKER_STATE["window"] = window
    _WORKER_STATE["elev"] = elev
    _WORKER_STATE["azim"] = azim
    _WORKER_STATE["azim_step"] = azim_step
    _WORKER_STATE["cmap"] = cmap
    _WORKER_STATE["z_scale"] = z_scale
    _WORKER_STATE["out_dir"] = out_dir
    _WORKER_STATE["dpi"] = dpi
    _WORKER_STATE["z_label"] = z_label


def _render_frame(payload: tuple[int, int]) -> None:
    frame_pos, frame_idx = payload
    data = _WORKER_STATE["data"]
    x = _WORKER_STATE["x"]
    z_low = _WORKER_STATE["z_low"]
    z_high = _WORKER_STATE["z_high"]
    window = _WORKER_STATE["window"]
    elev = _WORKER_STATE["elev"]
    azim = _WORKER_STATE["azim"]
    azim_step = _WORKER_STATE["azim_step"]
    cmap = _WORKER_STATE["cmap"]
    z_scale = _WORKER_STATE["z_scale"]
    out_dir = _WORKER_STATE["out_dir"]
    dpi = _WORKER_STATE["dpi"]
    z_label = _WORKER_STATE["z_label"]

    start = max(0, frame_idx - window + 1)
    y = np.arange(start, frame_idx + 1)
    x_grid, y_grid = np.meshgrid(x, y)
    z_raw = data[start : frame_idx + 1, :]
    z = apply_scale(z_raw, z_scale)

    fig = plt.figure(figsize=(11, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(
        x_grid,
        y_grid,
        z,
        cmap=cmap,
        linewidth=0,
        antialiased=False,
    )
    flat = z_raw.ravel()
    if flat.size:
        top_n = min(3, flat.size)
        top_idx = np.argpartition(flat, -top_n)[-top_n:]
        y_idx, x_idx = np.unravel_index(top_idx, z_raw.shape)
        peak_vals = apply_scale(z_raw[y_idx, x_idx], z_scale)
        ax.scatter(
            x[x_idx],
            y[y_idx],
            peak_vals,
            c="red",
            s=45,
            depthshade=False,
        )
    ax.set_title(f"Enciphered bins (frame {frame_idx})")
    ax.set_xlabel("bin_index")
    ax.set_ylabel("frame_index")
    ax.set_zlabel(z_label)
    ax.set_xlim(0, x[-1] if len(x) else 1)
    ax.set_ylim(start, max(frame_idx, start + 1))
    ax.set_zlim(z_low, z_high)
    ax.view_init(elev=elev, azim=azim + frame_pos * azim_step)

    frame_path = out_dir / f"frame_{frame_pos:05d}.png"
    fig.savefig(frame_path, dpi=dpi)
    plt.close(fig)


def apply_scale(values: np.ndarray, scale: str) -> np.ndarray:
    if scale == "sqrt":
        return np.sqrt(np.maximum(values, 0.0))
    if scale == "log":
        return np.log1p(np.maximum(values, 0.0))
    return values


def smooth_data(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return values
    window = int(window)
    pad_left = window // 2
    pad_right = window - 1 - pad_left
    padded = np.pad(values, ((pad_left, pad_right), (0, 0)), mode="edge")
    cumsum = np.cumsum(padded, axis=0, dtype=float)
    cumsum = np.vstack([np.zeros((1, values.shape[1])), cumsum])
    smoothed = (cumsum[window:] - cumsum[:-window]) / float(window)
    return smoothed


def render_video_parallel(
    data: np.ndarray,
    output: Path,
    fps: int,
    dpi: int,
    window: int,
    frame_step: int,
    max_frames: Optional[int],
    elev: float,
    azim: float,
    azim_step: float,
    metric: str,
    cmap: str,
    z_percentile_low: float,
    z_percentile_high: float,
    z_min: Optional[float],
    z_max: Optional[float],
    z_scale: str,
    smooth_window: int,
    workers: int,
) -> None:
    data_plot = smooth_data(data, smooth_window)
    frames_total, bin_count = data_plot.shape
    frame_indices = list(range(0, frames_total, max(1, frame_step)))
    if max_frames is not None:
        frame_indices = frame_indices[: max_frames]

    x = np.arange(bin_count)
    z_flat = data_plot.flatten() if data_plot.size else np.array([0.0])
    z_flat_scaled = apply_scale(z_flat, z_scale)
    if z_min is not None and z_max is not None:
        z_low = min(z_min, z_max)
        z_high = max(z_min, z_max)
    else:
        z_percentile_low = max(0.0, min(100.0, z_percentile_low))
        z_percentile_high = max(z_percentile_low, min(100.0, z_percentile_high))
        z_low = float(np.percentile(z_flat_scaled, z_percentile_low))
        z_high = float(np.percentile(z_flat_scaled, z_percentile_high))
    if z_high <= z_low:
        z_low = float(np.min(z_flat_scaled))
        z_high = float(np.max(z_flat_scaled))
    if z_high <= z_low:
        z_high = z_low + 1.0

    z_label = "match_pct" if metric == "float" else "match_count"
    frames_dir = output.with_suffix("")
    frames_dir = Path(f"{frames_dir}_frames")
    frames_dir.mkdir(parents=True, exist_ok=True)

    worker_count = workers if workers > 0 else cpu_count()
    payloads = list(enumerate(frame_indices))

    with Pool(
        processes=worker_count,
        initializer=_init_worker,
        initargs=(
            data_plot,
            x,
            z_low,
            z_high,
            window,
            elev,
            azim,
            azim_step,
            cmap,
            z_scale,
            frames_dir,
            dpi,
            z_label,
        ),
    ) as pool:
        for _ in pool.imap_unordered(_render_frame, payloads, chunksize=4):
            pass

    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-framerate",
                str(fps),
                "-i",
                "frame_%05d.png",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                str(output.resolve()),
            ],
            cwd=frames_dir,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        print(f"Wrote video to {output}")
        shutil.rmtree(frames_dir, ignore_errors=True)
    except Exception as exc:
        print(f"FFmpeg unavailable ({exc}); frames are in {frames_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a 3D scrolling video from enciphered_bins.csv"
    )
    parser.add_argument(
        "--input",
        default="enciphered_decryption_bins.csv",
        help="CSV input path",
    )
    parser.add_argument("--output", default="enciphered_bins.mp4", help="Video output path")
    parser.add_argument(
        "--metric",
        choices=["int", "float"],
        default="int",
        help="Use integer counts or normalized float counts",
    )
    parser.add_argument("--fps", type=int, default=24, help="Frames per second")
    parser.add_argument("--dpi", type=int, default=140, help="Output DPI")
    parser.add_argument("--window", type=int, default=60, help="Frames visible per view")
    parser.add_argument("--frame-step", type=int, default=1, help="Frame stride for animation")
    parser.add_argument("--max-frames", type=int, default=None, help="Limit number of frames")
    parser.add_argument("--elev", type=float, default=25.0, help="Camera elevation")
    parser.add_argument("--azim", type=float, default=-55.0, help="Camera azimuth")
    parser.add_argument(
        "--azim-step",
        type=float,
        default=0.0,
        help="Azimuth step per frame (0 disables rotation)",
    )
    parser.add_argument("--cmap", default="viridis", help="Matplotlib colormap")
    parser.add_argument(
        "--z-percentile-low",
        type=float,
        default=45.0,
        help="Lower percentile for z scaling",
    )
    parser.add_argument(
        "--z-percentile-high",
        type=float,
        default=55.0,
        help="Upper percentile for z scaling",
    )
    parser.add_argument(
        "--z-min",
        type=float,
        default=None,
        help="Absolute minimum for z scale (overrides percentiles)",
    )
    parser.add_argument(
        "--z-max",
        type=float,
        default=None,
        help="Absolute maximum for z scale (overrides percentiles)",
    )
    parser.add_argument(
        "--z-scale",
        choices=["linear", "sqrt", "log"],
        default="linear",
        help="Scale z values to emphasize differences",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=25,
        help="Temporal smoothing window for z values",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Render frames in parallel and stitch with ffmpeg",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Number of worker processes for parallel rendering (0 = cpu count)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"Input CSV not found: {input_path}", file=sys.stderr)
        return 1

    metric = "float" if args.metric == "float" else "int"
    data = load_bins(input_path, metric)
    if args.parallel:
        render_video_parallel(
            data=data,
            output=output_path,
            fps=args.fps,
            dpi=args.dpi,
            window=max(1, args.window),
            frame_step=max(1, args.frame_step),
            max_frames=args.max_frames,
            elev=args.elev,
            azim=args.azim,
            azim_step=args.azim_step,
            metric=metric,
            cmap=args.cmap,
            z_percentile_low=args.z_percentile_low,
            z_percentile_high=args.z_percentile_high,
            z_min=args.z_min,
            z_max=args.z_max,
            z_scale=args.z_scale,
            smooth_window=max(1, args.smooth_window),
            workers=args.workers,
        )
    else:
        render_video(
            data=data,
            output=output_path,
            fps=args.fps,
            dpi=args.dpi,
            window=max(1, args.window),
            frame_step=max(1, args.frame_step),
            max_frames=args.max_frames,
            elev=args.elev,
            azim=args.azim,
            azim_step=args.azim_step,
            metric=metric,
            cmap=args.cmap,
            z_percentile_low=args.z_percentile_low,
            z_percentile_high=args.z_percentile_high,
            z_min=args.z_min,
            z_max=args.z_max,
            z_scale=args.z_scale,
            smooth_window=max(1, args.smooth_window),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
