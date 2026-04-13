from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache


DEFAULT_BINARY_SYMBOLS: tuple[str, str] = (".", "-")


@dataclass(frozen=True)
class SequenceFeatures:
    word_count: int
    code_count: int
    mean_code_length: float
    same_length_run_count: int
    same_length_run_coverage: float
    mean_same_length_run_size: float
    max_same_length_run_size: int
    alternating_share: float


def validate_binary_symbols(symbols: tuple[str, str]) -> tuple[str, str]:
    if len(symbols) != 2:
        raise ValueError(f"binary symbols must contain exactly two entries: {symbols!r}")
    first, second = symbols
    if not isinstance(first, str) or not isinstance(second, str):
        raise TypeError("binary symbols must be strings")
    if len(first) != 1 or len(second) != 1 or first == second:
        raise ValueError(f"binary symbols must be two distinct single characters: {symbols!r}")
    return first, second


def split_sequence_words(sequence: str) -> list[list[str]]:
    if not isinstance(sequence, str):
        raise TypeError("sequence must be a str")

    sequence = sequence.strip()
    if not sequence:
        return []

    words: list[list[str]] = []
    for word in sequence.split("/"):
        codes = [code for code in word.strip().split() if code]
        if codes:
            words.append(codes)
    return words


def same_length_runs(codes: list[str]) -> list[list[str]]:
    if not codes:
        return []

    runs: list[list[str]] = []
    current = [codes[0]]
    for code in codes[1:]:
        if len(code) == len(current[-1]):
            current.append(code)
        else:
            runs.append(current)
            current = [code]
    runs.append(current)
    return runs


def singleton_runs(codes: list[str]) -> list[list[str]]:
    return [[code] for code in codes]


@lru_cache(maxsize=4096)
def alternating_identifier(code: str, symbols: tuple[str, str] = DEFAULT_BINARY_SYMBOLS) -> str | None:
    first, second = validate_binary_symbols(symbols)
    if len(code) < 2:
        return None
    if any(ch not in symbols for ch in code):
        return None
    if any(code[index] == code[index - 1] for index in range(1, len(code))):
        return None
    return "+" if code[0] == second else "-"


def normalized_sequence_length(text: str, sequence: str) -> int:
    word_count = len(text.split())
    boundary_count = max(word_count - 1, 0)
    return len(sequence) - (2 * boundary_count)


def extract_sequence_features(
    text: str,
    sequence: str,
    symbols: tuple[str, str] = DEFAULT_BINARY_SYMBOLS,
) -> SequenceFeatures:
    validate_binary_symbols(symbols)
    words = split_sequence_words(sequence)
    all_codes = [code for word in words for code in word]
    if not all_codes:
        return SequenceFeatures(
            word_count=0,
            code_count=0,
            mean_code_length=0.0,
            same_length_run_count=0,
            same_length_run_coverage=0.0,
            mean_same_length_run_size=0.0,
            max_same_length_run_size=0,
            alternating_share=0.0,
        )

    runs = [run for word in words for run in same_length_runs(word)]
    compressible_runs = [run for run in runs if len(run) >= 2]
    compressible_code_count = sum(len(run) for run in compressible_runs)
    alternating_count = sum(1 for code in all_codes if alternating_identifier(code, symbols) is not None)

    return SequenceFeatures(
        word_count=len(text.split()),
        code_count=len(all_codes),
        mean_code_length=sum(len(code) for code in all_codes) / len(all_codes),
        same_length_run_count=len(compressible_runs),
        same_length_run_coverage=compressible_code_count / len(all_codes),
        mean_same_length_run_size=(
            sum(len(run) for run in compressible_runs) / len(compressible_runs)
            if compressible_runs
            else 0.0
        ),
        max_same_length_run_size=max((len(run) for run in compressible_runs), default=0),
        alternating_share=alternating_count / len(all_codes),
    )
