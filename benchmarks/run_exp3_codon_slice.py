from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from benchmarks.run_exp3_codon_experiments import (  # noqa: E402
    DEFAULT_INPUT_TSV,
    DEFAULT_NATURAL_INPUT_TSV,
    DEFAULT_SHORT_MID_INPUT_TSV,
    DEFAULT_WINDOW_SIZE,
    DEFAULT_WINDOW_STRIDE,
    build_appendix_outputs,
    build_natural_distribution_appendix,
    build_outputs,
    evaluate_dataset,
    experiment_methods,
    filter_dataset_rows,
    load_dataset_rows,
    write_csv,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a single Exp3 codon slice.")
    parser.add_argument("--dataset", choices=("balanced", "natural", "short_mid"), required=True)
    parser.add_argument("--input-tsv", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--species", action="append", default=[])
    parser.add_argument("--length-group", action="append", default=[])
    parser.add_argument("--method", action="append", default=[])
    parser.add_argument("--time-repeat", type=int, default=1)
    parser.add_argument("--window-size", type=int, default=DEFAULT_WINDOW_SIZE)
    parser.add_argument("--stride", type=int, default=DEFAULT_WINDOW_STRIDE)
    args = parser.parse_args()

    default_input = {
        "balanced": DEFAULT_INPUT_TSV,
        "natural": DEFAULT_NATURAL_INPUT_TSV,
        "short_mid": DEFAULT_SHORT_MID_INPUT_TSV,
    }[args.dataset]
    input_tsv = args.input_tsv or default_input

    dataset_rows = filter_dataset_rows(
        load_dataset_rows(input_tsv),
        species_filter=set(args.species) if args.species else None,
        length_group_filter=set(args.length_group) if args.length_group else None,
    )
    if not dataset_rows:
        raise ValueError("no rows matched the requested species/length-group filters")

    methods = experiment_methods()
    requested_methods = set(args.method) if args.method else None
    if requested_methods:
        methods = [method for method in methods if method.name in requested_methods]
        if not methods:
            raise ValueError(f"no methods matched {sorted(requested_methods)}")

    available_methods = [method for method in methods if method.available]
    unavailable_methods = [method for method in methods if not method.available]

    sample_rows = evaluate_dataset(
        dataset_rows,
        methods=available_methods,
        time_repeat=max(args.time_repeat, 1),
        window_size=args.window_size,
        stride=args.stride,
        label=args.dataset,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.dataset == "balanced":
        build_outputs(sample_rows, args.output_dir)
    elif args.dataset == "natural":
        build_natural_distribution_appendix(sample_rows, args.output_dir)
    else:
        build_appendix_outputs(sample_rows, args.output_dir)

    unavailable_rows = [{"method": method.name, "note": method.note} for method in unavailable_methods]
    if unavailable_rows:
        write_csv(args.output_dir / "unavailable_methods.csv", unavailable_rows)
    print(f"Wrote: {args.output_dir}")


if __name__ == "__main__":
    main()
