from __future__ import annotations

import csv
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator


STARTS = {"ATG"}
STOPS = {"TAA", "TAG", "TGA"}
DNA_ALPHABET = {"A", "C", "G", "T"}
DEFAULT_WINDOW_SIZE = 8
DEFAULT_WINDOW_STRIDE = 8


@dataclass(frozen=True)
class CleanCdsRecord:
    species: str
    seq_id: str
    cds_nt: str
    has_internal_stop: bool


@dataclass(frozen=True)
class CodonFeatures:
    gc_content: float
    dominant_reference_coverage: float | None
    avg_window_heterogeneity: float | None
    sparse_difference_ratio: float | None


def parse_fasta_many(paths: Iterable[Path]) -> Iterator[tuple[str, str]]:
    for path in paths:
        seq_id: str | None = None
        parts: list[str] = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                if line.startswith(">"):
                    if seq_id is not None:
                        yield seq_id, "".join(parts)
                    seq_id = line[1:].split()[0]
                    parts = []
                    continue
                parts.append(line)
        if seq_id is not None:
            yield seq_id, "".join(parts)


def has_internal_stop(cds_nt: str) -> bool:
    codons = _chunk_codons(cds_nt)
    return any(codon in STOPS for codon in codons[1:-1])


def clean_records(
    records: Iterable[tuple[str, str]],
    *,
    species: str,
    min_nt_length: int = 90,
) -> list[CleanCdsRecord]:
    cleaned: list[CleanCdsRecord] = []
    seen_sequences: set[str] = set()
    for seq_id, sequence in records:
        cds_nt = "".join(sequence.upper().split())
        if not cds_nt or any(base not in DNA_ALPHABET for base in cds_nt):
            continue
        if len(cds_nt) % 3 != 0 or len(cds_nt) < min_nt_length:
            continue
        if cds_nt[:3] not in STARTS or cds_nt[-3:] not in STOPS:
            continue
        internal_stop = has_internal_stop(cds_nt)
        if internal_stop:
            continue
        if cds_nt in seen_sequences:
            continue
        seen_sequences.add(cds_nt)
        cleaned.append(
            CleanCdsRecord(
                species=species,
                seq_id=seq_id,
                cds_nt=cds_nt,
                has_internal_stop=internal_stop,
            )
        )
    return cleaned


def _chunk_codons(cds_nt: str) -> list[str]:
    if len(cds_nt) % 3 != 0:
        raise ValueError("CDS length must be divisible by 3")
    return [cds_nt[index : index + 3] for index in range(0, len(cds_nt), 3)]


def tokenize_cds(cds_nt: str, *, drop_terminal_stop: bool = True) -> list[str]:
    codons = _chunk_codons(cds_nt.upper())
    if drop_terminal_stop and codons and codons[-1] in STOPS:
        codons = codons[:-1]
    return codons


def codon_words(sequence: str) -> list[str]:
    compact = "".join(sequence.strip().split()).upper()
    if not compact:
        return []
    if len(compact) % 3 != 0:
        raise ValueError(f"invalid compact codon sequence length: {len(compact)}")
    return [compact[index : index + 3] for index in range(0, len(compact), 3)]


def normalize_codon_sequence(sequence: str) -> str:
    tokens = codon_words(sequence)
    for token in tokens:
        if len(token) != 3 or any(base not in DNA_ALPHABET for base in token):
            raise ValueError(f"invalid codon token: {token!r}")
    return "".join(tokens)


def normalized_codon_length(sequence: str) -> int:
    return len(normalize_codon_sequence(sequence))


def hamming_distance(left: str, right: str) -> int:
    if len(left) != len(right):
        raise ValueError("codons must have equal length")
    return sum(1 for left_base, right_base in zip(left, right) if left_base != right_base)


def _windows(tokens: list[str], *, window_size: int, stride: int) -> Iterator[list[str]]:
    if window_size <= 0 or stride <= 0:
        raise ValueError("window_size and stride must be positive")
    for start in range(0, len(tokens), stride):
        window = tokens[start : start + window_size]
        if window:
            yield window


def extract_codon_features(
    sequence: str,
    *,
    window_size: int = DEFAULT_WINDOW_SIZE,
    stride: int = DEFAULT_WINDOW_STRIDE,
) -> CodonFeatures:
    tokens = codon_words(sequence)
    bases = "".join(tokens)
    gc_content = (sum(1 for base in bases if base in {"G", "C"}) / len(bases)) if bases else 0.0
    if not tokens:
        return CodonFeatures(
            gc_content=gc_content,
            dominant_reference_coverage=None,
            avg_window_heterogeneity=None,
            sparse_difference_ratio=None,
        )

    coverage_values: list[float] = []
    heterogeneity_values: list[float] = []
    sparse_difference_values: list[float] = []
    for window in _windows(tokens, window_size=window_size, stride=stride):
        counts = Counter(window)
        reference, reference_count = min(counts.items(), key=lambda item: (-item[1], item[0]))
        non_reference = [token for token in window if token != reference]
        close_non_reference = [
            token for token in non_reference if hamming_distance(token, reference) <= 1
        ]
        coverage_values.append(reference_count / len(window))
        heterogeneity_values.append(len(counts) / len(window))
        sparse_difference_values.append(
            len(close_non_reference) / len(non_reference) if non_reference else 0.0
        )

    return CodonFeatures(
        gc_content=gc_content,
        dominant_reference_coverage=sum(coverage_values) / len(coverage_values),
        avg_window_heterogeneity=sum(heterogeneity_values) / len(heterogeneity_values),
        sparse_difference_ratio=sum(sparse_difference_values) / len(sparse_difference_values),
    )


def quantile(values: Iterable[float], q: float) -> float:
    if not 0.0 <= q <= 1.0:
        raise ValueError("q must be in [0, 1]")
    ordered = sorted(values)
    if not ordered:
        raise ValueError("quantile requires at least one value")
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def read_tsv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))
