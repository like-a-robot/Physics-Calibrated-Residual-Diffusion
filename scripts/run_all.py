"""Run the complete connected-zone temperature reconstruction experiment."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run(command):
    print("\n$", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def main(args):
    scripts = Path(__file__).resolve().parent
    common = ["--config", str(Path(args.config).resolve()), "--device", args.device]
    if args.smoke:
        common.append("--smoke")
    if args.resume:
        common.append("--resume")
    py = sys.executable

    commands = [] if args.postprocess_only else [
        [py, str(scripts / "00_prepare.py"), *common],
        [py, str(scripts / "01_zone_da.py"), *common, "--stage", "all"],
        [py, str(scripts / "02_autoencoder.py"), *common, "--target", "full", "--stage", "all"],
        [py, str(scripts / "02_autoencoder.py"), *common, "--target", "residual", "--stage", "all"],
        [py, str(scripts / "03_superresolution.py"), *common, "--method", "4dsrda"],
        [py, str(scripts / "03_superresolution.py"), *common, "--method", "diffsrda"],
        [py, str(scripts / "03_superresolution.py"), *common, "--method", "proposed"],
    ]
    commands += [
        [py, str(scripts / "04_reconstruct.py"), *common, "--method", "zone_const"],
        [py, str(scripts / "04_reconstruct.py"), *common, "--method", "4dsrda"],
        [py, str(scripts / "04_reconstruct.py"), *common, "--method", "diffsrda"],
        [py, str(scripts / "04_reconstruct.py"), *common, "--method", "proposed"],
        [py, str(scripts / "05_evaluate.py"), *common],
    ]
    if not args.skip_plots:
        commands.append([py, str(scripts / "08_plot_paper_slice_layout_4x3.py"), *common])
    if not args.skip_benchmark:
        benchmark = [py, str(scripts / "07_benchmark.py"), *common]
        if args.all_benchmark_frames:
            benchmark.append("--all-frames")
        commands.append(benchmark)

    for command in commands:
        run(command)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(Path(__file__).with_name("config.yaml")))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--postprocess-only", action="store_true",
        help="Run reconstruction, evaluation, paper plots, and benchmark using existing checkpoints and assimilation.",
    )
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--skip-benchmark", action="store_true")
    parser.add_argument(
        "--all-benchmark-frames",
        action="store_true",
        help="Benchmark all online frames once instead of estimating trajectory runtime from sampled frames.",
    )
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
