from __future__ import annotations

import argparse
import csv
import random
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from codon_utils import (  # noqa: E402
    DEFAULT_WINDOW_SIZE,
    DEFAULT_WINDOW_STRIDE,
    CleanCdsRecord,
    clean_records,
    extract_codon_features,
    parse_fasta_many,
    quantile,
    tokenize_cds,
)


DEFAULT_DATA_ROOT = ROOT_DIR.parent / "gene_dataset"
DEFAULT_OUTPUT_ROOT = ROOT_DIR / "exp3_real_codon"
DEFAULT_BALANCED_BIN_SAMPLE_SIZE = 150
DEFAULT_NATURAL_SPECIES_SAMPLE_SIZE = 1000
DEFAULT_SHORT_MID_BALANCED_BIN_SAMPLE_SIZE = 100
MIN_REGULARITY_BIN_N = 40
SPECIES_SOURCES: dict[str, dict[str, str]] = {
    "ecoli": {
        "dataset_dir": "e-coli_dataset",
        "raw_filename": "ecoli_cds.fna",
    },
    "human": {
        "dataset_dir": "homo_dataset",
        "raw_filename": "human_cds.fna",
    },
    "scerevisiae": {
        "dataset_dir": "s-cerevisiae_dataset",
        "raw_filename": "scerevisiae_cds.fna",
    },
}
MAIN_LENGTH_GROUPS: tuple[tuple[str, int, int | None], ...] = (
    ("L1_short", 30, 79),
    ("L2_medium", 80, 149),
    ("L3_long", 150, 299),
    ("L4_very_long", 300, None),
)
SHORT_MID_LENGTH_GROUPS: tuple[tuple[str, int, int | None], ...] = (
    ("L1_very_short", 6, 12),
    ("L2_short", 13, 20),
    ("L3_mid_short", 21, 40),
    ("L4_longer", 41, None),
)


@dataclass(frozen=True)
class SamplingProfile:
    name: str
    min_nt_length: int
    min_codons_after_drop_stop: int
    length_groups: tuple[tuple[str, int, int | None], ...]
    target_per_species_length_bin: int
    sampled_dirname: str
    tokenized_suffix: str


MAIN_PROFILE = SamplingProfile(
    name="main",
    min_nt_length=90,
    min_codons_after_drop_stop=30,
    length_groups=MAIN_LENGTH_GROUPS,
    target_per_species_length_bin=DEFAULT_BALANCED_BIN_SAMPLE_SIZE,
    sampled_dirname="sampled_balanced",
    tokenized_suffix="",
)
SHORT_MID_PROFILE = SamplingProfile(
    name="short_mid",
    min_nt_length=24,
    min_codons_after_drop_stop=6,
    length_groups=SHORT_MID_LENGTH_GROUPS,
    target_per_species_length_bin=DEFAULT_SHORT_MID_BALANCED_BIN_SAMPLE_SIZE,
    sampled_dirname="sampled_short_mid_balanced",
    tokenized_suffix="_short_mid",
)


def find_fasta_paths(dataset_dir: Path) -> list[Path]:
    return sorted(dataset_dir.rglob("cds_from_genomic.fna"))


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def stage_raw_fasta(target_path: Path, fasta_paths: list[Path]) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with target_path.open("wb") as target:
        first = True
        for fasta_path in fasta_paths:
            if not first:
                target.write(b"\n")
            with fasta_path.open("rb") as source:
                shutil.copyfileobj(source, target)
            target.write(b"\n")
            first = False


def record_to_cleaned_row(record: CleanCdsRecord) -> dict[str, Any]:
    return {
        "species": record.species,
        "seq_id": record.seq_id,
        "cds_nt": record.cds_nt,
        "has_internal_stop": str(record.has_internal_stop).lower(),
    }


def assign_length_group(num_codons: int, length_groups: tuple[tuple[str, int, int | None], ...]) -> str:
    for group_name, lower, upper in length_groups:
        if num_codons >= lower and (upper is None or num_codons <= upper):
            return group_name
    raise ValueError(f"num_codons is outside configured length groups: {num_codons}")


def build_tokenized_rows(
    cleaned_records: list[CleanCdsRecord],
    *,
    window_size: int,
    stride: int,
    drop_terminal_stop: bool,
    terminal_stop_policy: str,
    min_codons_after_drop_stop: int,
    length_groups: tuple[tuple[str, int, int | None], ...],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in cleaned_records:
        tokens = tokenize_cds(record.cds_nt, drop_terminal_stop=drop_terminal_stop)
        if len(tokens) < min_codons_after_drop_stop:
            continue

        token_sequence = " ".join(tokens)
        features = extract_codon_features(
            token_sequence,
            window_size=window_size,
            stride=stride,
        )
        rows.append(
            {
                "species": record.species,
                "seq_id": record.seq_id,
                "cds_nt": record.cds_nt,
                "codon_tokens": token_sequence,
                "num_codons": len(tokens),
                "gc_content": f"{features.gc_content:.6f}",
                "dominant_reference_coverage": (
                    "" if features.dominant_reference_coverage is None else f"{features.dominant_reference_coverage:.6f}"
                ),
                "avg_window_heterogeneity": (
                    "" if features.avg_window_heterogeneity is None else f"{features.avg_window_heterogeneity:.6f}"
                ),
                "sparse_difference_ratio": (
                    "" if features.sparse_difference_ratio is None else f"{features.sparse_difference_ratio:.6f}"
                ),
                "has_internal_stop": str(record.has_internal_stop).lower(),
                "length_group": assign_length_group(len(tokens), length_groups),
                "terminal_stop_policy": terminal_stop_policy,
            }
        )
    return rows


def sample_natural_rows(
    tokenized_rows: list[dict[str, Any]],
    *,
    sample_size_per_species: int,
    random_seed: int,
    species_order: list[str],
) -> list[dict[str, Any]]:
    rng = random.Random(random_seed)
    sampled: list[dict[str, Any]] = []
    for species in species_order:
        rows = sorted(
            [row for row in tokenized_rows if row["species"] == species],
            key=lambda row: str(row["seq_id"]),
        )
        if len(rows) > sample_size_per_species:
            indices = sorted(rng.sample(range(len(rows)), sample_size_per_species))
            sampled.extend(rows[index] for index in indices)
        else:
            sampled.extend(rows)
    return sampled


def sample_balanced_rows(
    tokenized_rows: list[dict[str, Any]],
    *,
    sample_size_per_species_length_bin: int,
    random_seed: int,
    species_order: list[str],
    length_groups: tuple[tuple[str, int, int | None], ...],
) -> list[dict[str, Any]]:
    rng = random.Random(random_seed)
    sampled: list[dict[str, Any]] = []
    for species in species_order:
        for length_group, _, _ in length_groups:
            rows = sorted(
                [
                    row
                    for row in tokenized_rows
                    if row["species"] == species and row["length_group"] == length_group
                ],
                key=lambda row: str(row["seq_id"]),
            )
            if len(rows) > sample_size_per_species_length_bin:
                indices = sorted(rng.sample(range(len(rows)), sample_size_per_species_length_bin))
                sampled.extend(rows[index] for index in indices)
            else:
                sampled.extend(rows)
    return sampled


def add_sampling_metadata(
    rows: list[dict[str, Any]],
    *,
    sampling_view: str,
) -> list[dict[str, Any]]:
    enriched = []
    for row in rows:
        updated = dict(row)
        updated["sampling_view"] = sampling_view
        enriched.append(updated)
    return enriched


def add_regularity_groups(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_bucket: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        by_bucket.setdefault((str(row["species"]), str(row["length_group"])), []).append(row)

    enriched: list[dict[str, Any]] = []
    for bucket_rows in by_bucket.values():
        regularity_values = [
            float(row["dominant_reference_coverage"])
            for row in bucket_rows
            if str(row.get("dominant_reference_coverage", "")).strip()
        ]
        bucket_n = len(bucket_rows)
        if not regularity_values:
            for row in bucket_rows:
                updated = dict(row)
                updated["regularity_group"] = "mid_regularity"
                updated["regularity_quartile"] = "q2"
                updated["regularity_bin_n"] = bucket_n
                updated["regularity_q1"] = ""
                updated["regularity_q3"] = ""
                updated["regularity_is_reportable"] = "false"
                enriched.append(updated)
            continue

        q1 = quantile(regularity_values, 0.25)
        q2 = quantile(regularity_values, 0.50)
        q3 = quantile(regularity_values, 0.75)
        is_reportable = bucket_n >= MIN_REGULARITY_BIN_N and q1 < q3

        for row in bucket_rows:
            regularity = float(row["dominant_reference_coverage"])
            updated = dict(row)
            if regularity <= q1:
                updated["regularity_group"] = "low_regularity"
            elif regularity >= q3:
                updated["regularity_group"] = "high_regularity"
            else:
                updated["regularity_group"] = "mid_regularity"

            if regularity <= q1:
                updated["regularity_quartile"] = "q1"
            elif regularity <= q2:
                updated["regularity_quartile"] = "q2"
            elif regularity <= q3:
                updated["regularity_quartile"] = "q3"
            else:
                updated["regularity_quartile"] = "q4"
            updated["regularity_bin_n"] = bucket_n
            updated["regularity_q1"] = f"{q1:.6f}"
            updated["regularity_q3"] = f"{q3:.6f}"
            updated["regularity_is_reportable"] = str(is_reportable).lower()
            enriched.append(updated)
    return sorted(enriched, key=lambda row: (row["species"], row["length_group"], row["seq_id"]))


def write_sample_slices(
    rows: list[dict[str, Any]],
    sampled_dir: Path,
    *,
    length_groups: tuple[tuple[str, int, int | None], ...],
) -> None:
    write_tsv(sampled_dir / "overall.tsv", rows)
    for length_group, _, _ in length_groups:
        write_tsv(
            sampled_dir / f"{length_group}.tsv",
            [row for row in rows if row["length_group"] == length_group],
        )
    write_tsv(
        sampled_dir / "high_regularity.tsv",
        [row for row in rows if row["regularity_group"] == "high_regularity"],
    )
    write_tsv(
        sampled_dir / "low_regularity.tsv",
        [row for row in rows if row["regularity_group"] == "low_regularity"],
    )


def write_length_diagnostics(
    rows: list[dict[str, Any]],
    tokenized_dir: Path,
    *,
    profile_name: str,
    species_order: list[str],
    length_groups: tuple[tuple[str, int, int | None], ...],
) -> None:
    group_counts: list[dict[str, Any]] = []
    for species in species_order:
        for length_group, _, _ in length_groups:
            group_counts.append(
                {
                    "profile": profile_name,
                    "species": species,
                    "length_group": length_group,
                    "samples": sum(
                        1
                        for row in rows
                        if row["species"] == species and row["length_group"] == length_group
                    ),
                }
            )

    histogram_counts: dict[tuple[str, int, int], int] = {}
    for row in rows:
        num_codons = int(row["num_codons"])
        bin_start = (num_codons // 50) * 50
        bin_end = bin_start + 49
        key = (str(row["species"]), bin_start, bin_end)
        histogram_counts[key] = histogram_counts.get(key, 0) + 1
    histogram_rows = [
        {
            "profile": profile_name,
            "species": species,
            "codon_count_bin_start": bin_start,
            "codon_count_bin_end": bin_end,
            "samples": samples,
        }
        for (species, bin_start, bin_end), samples in sorted(histogram_counts.items())
    ]

    write_tsv(tokenized_dir / f"length_group_counts_{profile_name}_drop_stop.tsv", group_counts)
    write_tsv(tokenized_dir / f"length_histogram_{profile_name}_drop_stop.tsv", histogram_rows)


def prepare_token_view(
    all_cleaned_records: list[CleanCdsRecord],
    *,
    tokenized_dir: Path,
    window_size: int,
    stride: int,
    drop_terminal_stop: bool,
    terminal_stop_policy: str,
    species_order: list[str],
    profile: SamplingProfile,
) -> list[dict[str, Any]]:
    all_tokenized_rows: list[dict[str, Any]] = []
    for species in species_order:
        species_records = [record for record in all_cleaned_records if record.species == species]
        tokenized_rows = build_tokenized_rows(
            species_records,
            window_size=window_size,
            stride=stride,
            drop_terminal_stop=drop_terminal_stop,
            terminal_stop_policy=terminal_stop_policy,
            min_codons_after_drop_stop=profile.min_codons_after_drop_stop,
            length_groups=profile.length_groups,
        )
        suffix = profile.tokenized_suffix if drop_terminal_stop else f"{profile.tokenized_suffix}_with_stop"
        write_tsv(tokenized_dir / f"{species}_tokens{suffix}.tsv", tokenized_rows)
        all_tokenized_rows.extend(tokenized_rows)
    return all_tokenized_rows


def write_sample_view(
    rows: list[dict[str, Any]],
    *,
    sampled_dir: Path,
    sampling_view: str,
    length_groups: tuple[tuple[str, int, int | None], ...],
    write_slices: bool = True,
) -> list[dict[str, Any]]:
    rows = add_sampling_metadata(rows, sampling_view=sampling_view)
    rows = add_regularity_groups(rows)
    if write_slices:
        write_sample_slices(rows, sampled_dir, length_groups=length_groups)
    else:
        write_tsv(sampled_dir / "overall.tsv", rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare the main and short-mid Exp3 codon benchmarks.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument(
        "--sample-size-per-species-length-bin",
        type=int,
        default=DEFAULT_BALANCED_BIN_SAMPLE_SIZE,
        help="Target per species-length bin for the main balanced benchmark.",
    )
    parser.add_argument(
        "--sample-size-per-species-natural",
        type=int,
        default=DEFAULT_NATURAL_SPECIES_SAMPLE_SIZE,
    )
    parser.add_argument(
        "--sample-size-per-species-length-bin-short-mid",
        type=int,
        default=DEFAULT_SHORT_MID_BALANCED_BIN_SAMPLE_SIZE,
        help="Target per species-length bin for the supplementary short-mid benchmark.",
    )
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--window-size", type=int, default=DEFAULT_WINDOW_SIZE)
    parser.add_argument("--stride", type=int, default=DEFAULT_WINDOW_STRIDE)
    parser.add_argument("--species", action="append", choices=sorted(SPECIES_SOURCES), default=[])
    args = parser.parse_args()

    profiles = (
        MAIN_PROFILE,
        SHORT_MID_PROFILE,
    )
    profile_sample_sizes = {
        MAIN_PROFILE.name: args.sample_size_per_species_length_bin,
        SHORT_MID_PROFILE.name: args.sample_size_per_species_length_bin_short_mid,
    }

    raw_dir = args.output_root / "raw"
    cleaned_dir = args.output_root / "cleaned"
    tokenized_dir = args.output_root / "tokenized"
    active_species = args.species or list(SPECIES_SOURCES)

    cleaned_records_by_profile: dict[str, list[CleanCdsRecord]] = {profile.name: [] for profile in profiles}
    for species in active_species:
        config = SPECIES_SOURCES[species]
        dataset_dir = args.data_root / config["dataset_dir"]
        fasta_paths = find_fasta_paths(dataset_dir)
        if not fasta_paths:
            raise FileNotFoundError(f"no cds_from_genomic.fna found under {dataset_dir}")

        stage_raw_fasta(raw_dir / config["raw_filename"], fasta_paths)
        parsed_records = list(parse_fasta_many(fasta_paths))
        for profile in profiles:
            cleaned_records = clean_records(
                parsed_records,
                species=species,
                min_nt_length=profile.min_nt_length,
            )
            cleaned_records_by_profile[profile.name].extend(cleaned_records)
            cleaned_suffix = "" if profile.name == MAIN_PROFILE.name else f"_{profile.name}"
            write_tsv(
                cleaned_dir / f"{species}_cleaned{cleaned_suffix}.tsv",
                [record_to_cleaned_row(record) for record in cleaned_records],
            )
            print(
                f"[prepared] profile={profile.name} species={species}: "
                f"cleaned={len(cleaned_records)} raw_fastas={len(fasta_paths)}"
            )

    # Main benchmark artifacts.
    main_drop_stop_rows = prepare_token_view(
        cleaned_records_by_profile[MAIN_PROFILE.name],
        tokenized_dir=tokenized_dir,
        window_size=args.window_size,
        stride=args.stride,
        drop_terminal_stop=True,
        terminal_stop_policy="drop_stop",
        species_order=active_species,
        profile=MAIN_PROFILE,
    )
    main_keep_stop_rows = prepare_token_view(
        cleaned_records_by_profile[MAIN_PROFILE.name],
        tokenized_dir=tokenized_dir,
        window_size=args.window_size,
        stride=args.stride,
        drop_terminal_stop=False,
        terminal_stop_policy="keep_stop",
        species_order=active_species,
        profile=MAIN_PROFILE,
    )
    write_length_diagnostics(
        main_drop_stop_rows,
        tokenized_dir,
        profile_name=MAIN_PROFILE.name,
        species_order=active_species,
        length_groups=MAIN_PROFILE.length_groups,
    )

    main_balanced_rows = sample_balanced_rows(
        main_drop_stop_rows,
        sample_size_per_species_length_bin=profile_sample_sizes[MAIN_PROFILE.name],
        random_seed=args.random_seed,
        species_order=active_species,
        length_groups=MAIN_PROFILE.length_groups,
    )
    natural_rows = sample_natural_rows(
        main_drop_stop_rows,
        sample_size_per_species=args.sample_size_per_species_natural,
        random_seed=args.random_seed,
        species_order=active_species,
    )
    main_balanced_with_stop_rows = sample_balanced_rows(
        main_keep_stop_rows,
        sample_size_per_species_length_bin=profile_sample_sizes[MAIN_PROFILE.name],
        random_seed=args.random_seed,
        species_order=active_species,
        length_groups=MAIN_PROFILE.length_groups,
    )

    main_balanced_rows = write_sample_view(
        main_balanced_rows,
        sampled_dir=args.output_root / MAIN_PROFILE.sampled_dirname,
        sampling_view="balanced",
        length_groups=MAIN_PROFILE.length_groups,
    )
    natural_rows = write_sample_view(
        natural_rows,
        sampled_dir=args.output_root / "sampled_natural",
        sampling_view="natural",
        length_groups=MAIN_PROFILE.length_groups,
    )
    main_balanced_with_stop_rows = write_sample_view(
        main_balanced_with_stop_rows,
        sampled_dir=args.output_root / "sampled_balanced_with_stop",
        sampling_view="balanced",
        length_groups=MAIN_PROFILE.length_groups,
        write_slices=False,
    )

    # Supplementary short-mid benchmark artifacts.
    short_mid_drop_stop_rows = prepare_token_view(
        cleaned_records_by_profile[SHORT_MID_PROFILE.name],
        tokenized_dir=tokenized_dir,
        window_size=args.window_size,
        stride=args.stride,
        drop_terminal_stop=True,
        terminal_stop_policy="drop_stop",
        species_order=active_species,
        profile=SHORT_MID_PROFILE,
    )
    write_length_diagnostics(
        short_mid_drop_stop_rows,
        tokenized_dir,
        profile_name=SHORT_MID_PROFILE.name,
        species_order=active_species,
        length_groups=SHORT_MID_PROFILE.length_groups,
    )
    short_mid_balanced_rows = sample_balanced_rows(
        short_mid_drop_stop_rows,
        sample_size_per_species_length_bin=profile_sample_sizes[SHORT_MID_PROFILE.name],
        random_seed=args.random_seed,
        species_order=active_species,
        length_groups=SHORT_MID_PROFILE.length_groups,
    )
    short_mid_balanced_rows = write_sample_view(
        short_mid_balanced_rows,
        sampled_dir=args.output_root / SHORT_MID_PROFILE.sampled_dirname,
        sampling_view="balanced",
        length_groups=SHORT_MID_PROFILE.length_groups,
    )

    print(f"[sampled_balanced] total={len(main_balanced_rows)} -> {args.output_root / MAIN_PROFILE.sampled_dirname}")
    print(f"[sampled_natural] total={len(natural_rows)} -> {args.output_root / 'sampled_natural'}")
    print(
        "[sampled_balanced_with_stop] "
        f"total={len(main_balanced_with_stop_rows)} -> {args.output_root / 'sampled_balanced_with_stop'}"
    )
    print(
        "[sampled_short_mid_balanced] "
        f"total={len(short_mid_balanced_rows)} -> {args.output_root / SHORT_MID_PROFILE.sampled_dirname}"
    )


if __name__ == "__main__":
    main()
